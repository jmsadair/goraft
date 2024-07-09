package raft

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/jmsadair/raft/internal/numeric"
	"github.com/jmsadair/raft/internal/random"
	"github.com/jmsadair/raft/logging"
)

var (
	// ErrNotLeader is returned when an operation or configuration change is
	// submitted to a node that is not a leader. Operations may only be submitted
	// to a node that is a leader.
	ErrNotLeader = errors.New("this node is not the leader")

	// ErrInvalidLease is returned when a lease-based read-only operation is
	// rejected due to the leader's lease having expired. The operation likely
	// needs to be submitted to a different node in the cluster.
	ErrInvalidLease = errors.New("this node does not have a valid lease")

	// ErrPendingConfiguration is returned when a membership change is requested and there
	// is a pending membership change that has not yet been committed. The membership change
	// request may be submitted once the pending membership change is committed.
	ErrPendingConfiguration = errors.New("only one membership change may be committed at a time")

	// ErrNoCommitThisTerm is returned when a membership change is requested but a log entry has yet
	// to be committed in the current term. The membership change may be submitted once a log entry
	// has been committed this term.
	ErrNoCommitThisTerm = errors.New("a log entry has not been committed in this term")
)

// The default chunk size for InstallSnapshot RPCs.
const snapshotChunkSize = 32 * 1024

// State represents the current state of a node.
// A node may either be shutdown, the leader, or a followers.
type State uint32

const (
	// Leader is a state indicating that the node is responsible for replicating and
	// committing log entries. The leader will accept operations and membership change
	// requests.
	Leader State = iota

	// Follower is a state indicating that a node is responsible for accepting log entries replicated
	// by the leader. A node in the follower state will not accept operations or membership change
	// requests.
	Follower

	// PreCandidate is a state indicating this node is holding a prevote. If this node is able to
	// receive successful RequestVote RPC responses from the majority of the cluster, it will enter the
	// candidate state.
	PreCandidate

	// Candidate is a state indicating that this node is currently holding an election. A node will
	// remain in this state until it is either elected the leader or another node is elected leader
	// thereby causing this node to step down to the follower state.
	Candidate

	// Shutdown is a state indicating that the node is currently offline.
	Shutdown
)

// String converts a State into a string.
func (s State) String() string {
	switch s {
	case Leader:
		return "leader"
	case Follower:
		return "follower"
	case Candidate:
		return "candidate"
	case Shutdown:
		return "shutdown"
	default:
		panic("invalid state")
	}
}

// Status is the status of a node.
type Status struct {
	// The unique identifier of this node.
	ID string

	// The address of the node.
	Address string

	// The current term.
	Term uint64

	// The current commit index.
	CommitIndex uint64

	// The index of the last log entry applied to the state machine.
	LastApplied uint64

	// The current state of the node: leader, followers, shutdown.
	State State
}

// follower contains all state associated with followers.
type follower struct {
	// The next log index that should be sent to this node.
	nextIndex uint64

	// The highest log index known to be replicated on this node.
	matchIndex uint64

	// The snapshot file to read when sending a snapshot to this node.
	snapshot SnapshotFile
}

// Raft implements the raft consensus protocol.
type Raft struct {
	// The ID of this node.
	id string

	// The address of this node.
	address string

	// The ID that this raft node believes is the leader. Used to redirect clients.
	leaderID string

	// The configuration options for this raft node.
	options options

	// The logger for this raft node.
	logger *logging.Logger

	// The network transport for sending and receiving RPCs.
	transport Transport

	// The latest configuration of the cluster.
	// This may or may not be committed.
	configuration *Configuration

	// The most recently committed configuration of the cluster.
	committedConfiguration *Configuration

	// A channel used to respond to membership change requests.
	configurationResponseCh chan Result[Configuration]

	// Maps ID to the state of the other nodes in the cluster.
	// Maintained by the leader.
	followers map[string]*follower

	// Manages both read-only and replicated operations.
	operationManager *operationManager

	// This stores and retrieves persisted log entries.
	log Log

	// This stores and retrieves persisted vote and term.
	stateStorage StateStorage

	// This stores and retrieves persisted snapshots.
	snapshotStorage SnapshotStorage

	// A writer for a snapshot file if one is being installed.
	snapshot SnapshotFile

	// The state machine provided by the client that operations will be applied to.
	fsm StateMachine

	// Notifies the apply loop that the commit index has been updated and that
	// replicated operation may be applied to the state machine.
	applyCond *sync.Cond

	// Notifies the commit loop that new log entries may be ready to be committed.
	commitCond *sync.Cond

	// Notifies read-only loop that read-only operations may be able to be applied
	// to the state machine.
	readOnlyCond *sync.Cond

	// Notifies election loop to start an election.
	electionCond *sync.Cond

	// Notifies snapshot loop that a snapshot should be taken.
	snapshotCond *sync.Cond

	// The current state of this raft node: leader, followers, or shutdown.
	state State

	// Index of the last log entry that was committed.
	commitIndex uint64

	// Index of the last log entry that was applied.
	lastApplied uint64

	// The current term of this raft node. Must be persisted.
	currentTerm uint64

	// The last included index of the most recent snapshot.
	lastIncludedIndex uint64

	// The last included term of the most recent snapshot.
	lastIncludedTerm uint64

	// ID of the candidate that this raft node voted for. Must be persisted.
	votedFor string

	// The timestamp representing the time of the last contact by the leader.
	lastContact time.Time

	wg sync.WaitGroup

	mu sync.Mutex
}

// NewRaft creates a new instance of Raft with the provided ID and address.
// The datapath is the top level directory where all state for this node will be persisted.
func NewRaft(
	id string,
	address string,
	fsm StateMachine,
	dataPath string,
	opts ...Option,
) (*Raft, error) {
	// Apply provided options.
	var options options
	for _, opt := range opts {
		if err := opt(&options); err != nil {
			return nil, err
		}
	}

	// Set default values if option not provided.
	if !options.levelSet {
		options.logLevel = logging.Info
	}
	if options.heartbeatInterval == 0 {
		options.heartbeatInterval = defaultHeartbeat
	}
	if options.electionTimeout == 0 {
		options.electionTimeout = defaultElectionTimeout
	}
	if options.leaseDuration == 0 {
		options.leaseDuration = defaultLeaseDuration
	}
	if options.log == nil {
		log, err := NewLog(dataPath)
		if err != nil {
			return nil, err
		}
		options.log = log
	}
	if options.stateStorage == nil {
		stateStore, err := NewStateStorage(dataPath)
		if err != nil {
			return nil, err
		}
		options.stateStorage = stateStore
	}
	if options.snapshotStorage == nil {
		snapshotStore, err := NewSnapshotStorage(dataPath)
		if err != nil {
			return nil, err
		}
		options.snapshotStorage = snapshotStore
	}
	if options.transport == nil {
		transport, err := NewTransport(address)
		if err != nil {
			return nil, err
		}
		options.transport = transport
	}

	logger, err := logging.NewLogger(logging.WithLevel(options.logLevel))
	if err != nil {
		return nil, err
	}

	raft := &Raft{
		id:               id,
		address:          address,
		logger:           logger,
		log:              options.log,
		stateStorage:     options.stateStorage,
		snapshotStorage:  options.snapshotStorage,
		transport:        options.transport,
		options:          options,
		operationManager: newOperationManager(options.leaseDuration),
		state:            Shutdown,
		fsm:              fsm,
	}

	raft.applyCond = sync.NewCond(&raft.mu)
	raft.commitCond = sync.NewCond(&raft.mu)
	raft.readOnlyCond = sync.NewCond(&raft.mu)
	raft.electionCond = sync.NewCond(&raft.mu)
	raft.snapshotCond = sync.NewCond(&raft.mu)

	if err := raft.restore(); err != nil {
		return nil, err
	}

	return raft, nil
}

// restore will recover any persisted state and initialize the node with it.
func (r *Raft) restore() error {
	if err := r.log.Open(); err != nil {
		return fmt.Errorf("could not open log: %w", err)
	}
	if err := r.log.Replay(); err != nil {
		return fmt.Errorf("could not replay log: %w", err)
	}

	// Restore the current term and vote.
	currentTerm, votedFor, err := r.stateStorage.State()
	if err != nil {
		return fmt.Errorf("could not recover state from storage: %w", err)
	}
	r.currentTerm = currentTerm
	r.votedFor = votedFor

	// Restore the state machine from the most recent snapshot.
	file, err := r.snapshotStorage.SnapshotFile()
	if err != nil {
		return fmt.Errorf("could not get snapshot file: %w", err)
	}
	if file != nil {
		metadata := file.Metadata()
		r.lastIncludedIndex = metadata.LastIncludedIndex
		r.lastIncludedTerm = metadata.LastIncludedTerm
		r.commitIndex = metadata.LastIncludedIndex
		r.lastApplied = metadata.LastIncludedIndex
		if err := r.fsm.Restore(file); err != nil {
			return fmt.Errorf("could not restore state machine with snapshot: %w", err)
		}
		configuration, err := r.transport.DecodeConfiguration(metadata.Configuration)
		if err != nil {
			return fmt.Errorf("could not decode snapshot configuration: %w", err)
		}
		r.configuration = &configuration
		r.committedConfiguration = &configuration
		if err := file.Close(); err != nil {
			return fmt.Errorf("could not close snapshot file: %w", err)
		}
	}

	// Use the most recent configuration from the log.
	for index := r.lastIncludedIndex + 1; index <= r.log.LastIndex(); index++ {
		entry, err := r.log.GetEntry(index)
		if err != nil {
			return fmt.Errorf("could not get entry from log: %w", err)
		}
		if entry.EntryType != ConfigurationEntry {
			continue
		}
		configuration, err := r.transport.DecodeConfiguration(entry.Data)
		if err != nil {
			return fmt.Errorf("could not decode log entry configuration: %w", err)
		}
		if r.configuration != nil {
			committedConfiguration := r.configuration.Clone()
			r.committedConfiguration = &committedConfiguration
		}
		r.configuration = &configuration
	}

	return nil
}

// Bootstrap initializes this node with a cluster configuration.
// The configuration must contain the ID and address of all nodes
// in the cluster including this one.
//
// This function should only be called when starting a cluster for
// the first time and there is no existing configuration. This should
// only be called on a single, voting member of the cluster.
func (r *Raft) Bootstrap(configuration map[string]string) error {
	if _, ok := configuration[r.id]; !ok {
		return errors.New("configuration must contain this node")
	}
	if r.configuration != nil {
		return fmt.Errorf(
			"node already has an existing configuration: Bootstrap should only be called for a cluster starting for the first time: configuration = %s",
			r.configuration.String(),
		)
	}
	if r.log.LastIndex() > 0 {
		return errors.New(
			"node already has existing state: Bootstrap should only be called for a cluster starting for the first time",
		)
	}

	index := uint64(1)
	term := uint64(1)
	r.configuration = NewConfiguration(index, configuration)

	// Append the configuration to the log.
	data, err := r.transport.EncodeConfiguration(r.configuration)
	if err != nil {
		return err
	}
	entry := NewLogEntry(index, term, data, ConfigurationEntry)
	if err := r.log.AppendEntry(entry); err != nil {
		return err
	}

	r.logger.Infof("node bootstrapped: configuration = %s", r.configuration.String())

	return nil
}

// Start starts this node if it has not already been started.
//
// If the node does not have an existing configuration, a configuration
// will be created that only includes this node as a non-voter. If the node
// has been started before, Restart should be called instead.
func (r *Raft) Start() error {
	return r.start(false)
}

// Restart starts a node that has been started and stopped before.
//
// The difference between Restart and Start is that Restart restores
// the state of the node from non-volatile whereas Start does not. Restart
// should not be called if the node is being started for the first time.
func (r *Raft) Restart() error {
	return r.start(true)
}

// start will start this node if it is not already started. If restore is true,
// then any persisted state will be restored.
func (r *Raft) start(restore bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state != Shutdown {
		return nil
	}

	if restore {
		if err := r.restore(); err != nil {
			return fmt.Errorf("could not restore state: %w", err)
		}
	}

	if r.configuration == nil {
		r.configuration = &Configuration{}
		r.logger.Infof("node has no configuration, will start as non-voter")
	}

	// Register the functions for handling RPCs.
	r.transport.RegisterAppendEntriesHandler(r.AppendEntries)
	r.transport.RegisterRequestVoteHandler(r.RequestVote)
	r.transport.RegsiterInstallSnapshotHandler(r.InstallSnapshot)

	// Initialize the follower state.
	r.followers = make(map[string]*follower)
	for id := range r.configuration.Members {
		r.followers[id] = new(follower)
	}

	r.lastContact = time.Now()
	r.state = Follower

	r.wg.Add(7)
	go r.readOnlyLoop()
	go r.applyLoop()
	go r.electionTicker()
	go r.electionLoop()
	go r.heartbeatLoop()
	go r.commitLoop()
	go r.snapshotLoop()

	// Start serving incoming RPCs.
	if err := r.transport.Run(); err != nil {
		return fmt.Errorf("could not run transport: %w", err)
	}

	r.logger.Infof(
		"node started: address = %s electionTimeout = %v, heartbeatInterval = %v, leaseDuration = %v",
		r.address,
		r.options.electionTimeout,
		r.options.heartbeatInterval,
		r.options.leaseDuration,
	)

	return nil
}

// Stop stops this node if is not already stopped.
func (r *Raft) Stop() {
	r.mu.Lock()

	if r.state == Shutdown {
		r.mu.Unlock()
		return
	}

	r.state = Shutdown
	r.applyCond.Broadcast()
	r.commitCond.Broadcast()
	r.readOnlyCond.Broadcast()
	r.electionCond.Broadcast()
	r.snapshotCond.Broadcast()

	r.mu.Unlock()
	r.wg.Wait()

	// Stop accepting RPCs.
	r.transport.Shutdown()

	if err := r.log.Close(); err != nil {
		r.logger.Errorf("failed to close log: %v", err)
	}

	// Close or discard of any snapshot files.
	r.resetSnapshotFiles()

	r.logger.Info("node stopped")
}

// Status returns the status of this node. The status includes
// the ID, address, term, commit index, last applied index, and
// state of this node.
func (r *Raft) Status() Status {
	r.mu.Lock()
	defer r.mu.Unlock()

	return Status{
		ID:          r.id,
		Address:     r.transport.Address(),
		Term:        r.currentTerm,
		CommitIndex: r.commitIndex,
		LastApplied: r.lastApplied,
		State:       r.state,
	}
}

// Configuration returns the most current configuration of this node.
// This configuration may or may not have been committed yet. If there
// is no configuration, an empty configuration is returned.
func (r *Raft) Configuration() Configuration {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.configuration == nil {
		return Configuration{}
	}
	return r.configuration.Clone()
}

// AddServer will add a node with the provided ID and address to the cluster
// and return a future for the resulting configuration. It is generally recommended
// to add a new node as a non-voting member before adding it as a voting member so that
// it can become synced with the rest of the cluster.
//
// The provided ID must be unique from the existing nodes in the cluster.
// If the configuration change was not successful, the returned future will
// be populated with an error. It may be necessary to resubmit the configuration
// change to this node or a different node. It is safe to call this function as
// many times as necessary.
//
// It is is the caller's responsibility to implement retry logic.
func (r *Raft) AddServer(
	id string,
	address string,
	isVoter bool,
	timeout time.Duration,
) Future[Configuration] {
	r.mu.Lock()
	defer r.mu.Unlock()

	configurationFuture := newFuture[Configuration](timeout)

	// Only the leader can make membership changes.
	if r.state != Leader {
		respond(configurationFuture.responseCh, Configuration{}, ErrNotLeader)
		return configurationFuture
	}

	// Membership changes may not be submitted until a log entry for this term is committed.
	if !r.committedThisTerm() {
		respond(configurationFuture.responseCh, Configuration{}, ErrNoCommitThisTerm)
		return configurationFuture
	}

	// The membership change is still pending - wait until it completes.
	if r.pendingConfigurationChange() {
		respond(configurationFuture.responseCh, Configuration{}, ErrPendingConfiguration)
		return configurationFuture
	}

	// The provided node is already a part of the cluster.
	if r.isMember(id) && r.isVoter(id) == isVoter {
		respond(configurationFuture.responseCh, *r.configuration, nil)
		return configurationFuture
	}

	// Create the configuration that includes the new node as a non-voter.
	configuration := r.configuration.Clone()
	configuration.Members[id] = address
	configuration.IsVoter[id] = isVoter

	// Add the configuration to the log.
	r.appendConfiguration(&configuration)

	r.configuration = &configuration
	r.followers[id] = &follower{nextIndex: 1}

	r.sendAppendEntriesToPeers()

	r.logger.Debugf(
		"request to add node submitted: id = %s, address = %s, voter = %t, logIndex = %d",
		id,
		address,
		isVoter,
		configuration.Index,
	)

	return configurationFuture
}

// RemoveServer will remove the node with the provided ID from the cluster
// and returns a future for the resulting configuration. Once removed, the node
// will remain online as a non-voter and may safely be shutdown.
//
// If the configuration change was not successful, the returned future will be populated
// with an error. It may be necessary to resubmit the configuration change to this node
// or a different node. It is safe to call this function as many times as necessary.
//
// It is the caller's responsibility to implement retry logic.
func (r *Raft) RemoveServer(id string, timeout time.Duration) Future[Configuration] {
	r.mu.Lock()
	defer r.mu.Unlock()

	configurationFuture := newFuture[Configuration](timeout)

	// Only the leader can make membership changes.
	if r.state != Leader {
		respond(configurationFuture.responseCh, Configuration{}, ErrNotLeader)
		return configurationFuture
	}

	// Membership changes may not be submitted until a log entry for this term is committed.
	if !r.committedThisTerm() {
		respond(configurationFuture.responseCh, Configuration{}, ErrNoCommitThisTerm)
		return configurationFuture
	}

	// The membership change is still pending - wait until it completes.
	if r.pendingConfigurationChange() {
		respond(configurationFuture.responseCh, Configuration{}, ErrPendingConfiguration)
		return configurationFuture
	}

	// The provided node is already removed from the cluster.
	if !r.isMember(id) {
		respond(configurationFuture.responseCh, *r.configuration, nil)
		return configurationFuture
	}

	// Create the configuration without the node.
	configuration := r.configuration.Clone()
	delete(configuration.Members, id)
	delete(configuration.IsVoter, id)

	// Add the configuration to the log.
	r.appendConfiguration(&configuration)

	r.sendAppendEntriesToPeers()

	r.logger.Debugf(
		"request to remove node submitted: id = %s, logIndex = %d",
		id,
		configuration.Index,
	)

	return configurationFuture
}

// SubmitOperation accepts an operation for application to the state machine and returns a
// future for the response to the operation. Once the operation has been applied to the state
// machine, the returned future will be populated with the response.
//
// Even if the operation is submitted successfully, is not guaranteed that it will be applied
// to the state machine if there are failures. If the operation was unable to be applied
// to the state machine or the operation times out, the future will be populated with
// an error.
//
// It may be necessary to resubmit the operation to this node or a different node if it failed.
// It is the caller's responsibility to implement retry logic and to handle duplicate operations.
func (r *Raft) SubmitOperation(
	operation []byte,
	operationType OperationType,
	timeout time.Duration,
) Future[OperationResponse] {
	switch operationType {
	case Replicated:
		return r.submitReplicatedOperation(operation, timeout)
	case LeaseBasedReadOnly, LinearizableReadOnly:
		return r.submitReadOnlyOperation(operation, operationType, timeout)
	default:
		operationFuture := newFuture[OperationResponse](timeout)
		respond(
			operationFuture.responseCh,
			OperationResponse{},
			errors.New("operation type is not valid"),
		)
		return operationFuture
	}
}

// submitReplicatedOperation submits a replicated operation to be applied to the state machine.
func (r *Raft) submitReplicatedOperation(
	operationBytes []byte,
	timeout time.Duration,
) Future[OperationResponse] {
	r.mu.Lock()
	defer r.mu.Unlock()

	operationFuture := newFuture[OperationResponse](timeout)

	if r.state != Leader {
		respond(operationFuture.responseCh, OperationResponse{}, ErrNotLeader)
		return operationFuture
	}

	entry := NewLogEntry(r.log.NextIndex(), r.currentTerm, operationBytes, OperationEntry)
	if err := r.log.AppendEntry(entry); err != nil {
		r.logger.Fatalf("failed to append entry to log: error = %v", err)
	}

	r.operationManager.pendingReplicated[entry.Index] = operationFuture.responseCh

	r.sendAppendEntriesToPeers()

	r.logger.Debugf(
		"operation submitted: logIndex = %d, logTerm = %d, type = %s",
		entry.Index,
		entry.Term,
		Replicated.String(),
	)

	return operationFuture
}

// submitReadOnlyOperation submits a read-only operation to be applied to the state machine.
func (r *Raft) submitReadOnlyOperation(
	operationBytes []byte,
	readOnlyType OperationType,
	timeout time.Duration,
) Future[OperationResponse] {
	r.mu.Lock()
	defer r.mu.Unlock()

	operationFuture := newFuture[OperationResponse](timeout)

	if r.state != Leader {
		respond(operationFuture.responseCh, OperationResponse{}, ErrNotLeader)
		return operationFuture
	}

	operation := &Operation{
		Bytes:         operationBytes,
		OperationType: readOnlyType,
		readIndex:     r.commitIndex,
	}
	r.operationManager.pendingReadOnly[operation] = operationFuture.responseCh

	// If the last applied index is at least as the large as the read index, the
	// lease-based read can be served immediately.
	if readOnlyType == LeaseBasedReadOnly && operation.readIndex <= r.lastApplied {
		r.readOnlyCond.Broadcast()
	}

	// Linearizable read-only operations are served in batches. Verify quorum if is
	// is not already being verified.
	if readOnlyType == LinearizableReadOnly && r.operationManager.shouldVerifyQuorum {
		r.sendAppendEntriesToPeers()
		r.operationManager.shouldVerifyQuorum = false
	}

	r.logger.Debugf(
		"operation submitted: readIndex = %d, type = %s",
		operation.readIndex,
		operation.OperationType.String(),
	)

	return operationFuture
}

// AppendEntries handles log replication requests from the leader. It takes a request to append
// entries and fills the response with the result of the append operation. This will return an error
// if the node is shutdown.
func (r *Raft) AppendEntries(request *AppendEntriesRequest, response *AppendEntriesResponse) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state == Shutdown {
		return fmt.Errorf("could not execute RequestVote RPC: %s is shutdown", r.id)
	}

	r.logger.Debugf(
		"AppendEntries RPC received: leaderID = %s, leaderCommit = %d, term = %d, prevLogIndex = %d, prevLogTerm = %d",
		request.LeaderID,
		request.LeaderCommit,
		request.Term,
		request.PrevLogIndex,
		request.PrevLogTerm,
	)

	response.Term = r.currentTerm
	response.Success = false

	// Reject any requests with an out-of-date term.
	if request.Term < r.currentTerm {
		r.logger.Debugf(
			"AppendEntries RPC rejected: reason = out of date term, localTerm = %d, remoteTerm = %d",
			r.currentTerm,
			request.Term,
		)
		return nil
	}

	// Update the time of last contact - note that this should be done even
	// if the request is rejected due to having a non-matching previous log entry.
	r.lastContact = time.Now()

	// Update the ID of the node that this node recognizes as the leader.
	r.leaderID = request.LeaderID

	// If the request has a more up-to-date term, update current term and
	// become a followers.
	if request.Term > r.currentTerm {
		r.becomeFollower(request.LeaderID, request.Term)
		response.Term = r.currentTerm
	}

	// Transition to follower state if this node is still in the candidate or precandidate state.
	if request.Term == r.currentTerm && (r.state == Candidate || r.state == PreCandidate) {
		r.becomeFollower(request.LeaderID, request.Term)
	}

	// Reject the request if the log has been compacted and no longer contains the previous log entry.
	if r.lastIncludedIndex > request.PrevLogIndex {
		r.logger.Debugf(
			"AppendEntries RPC rejected: reason = log no longer contains previous entry, logIndex = %d, lastIncludedIndex = %d",
			request.PrevLogIndex,
			r.lastIncludedIndex,
		)
		response.Index = r.lastIncludedIndex + 1
		return nil
	}

	// Reject the request if the log is too short to contain the previous log entry.
	if r.log.NextIndex() <= request.PrevLogIndex {
		r.logger.Debugf(
			"AppendEntries RPC rejected: reason = log does not contain previous entry, logIndex = %d, lastLogIndex = %d",
			request.PrevLogIndex,
			r.log.LastIndex(),
		)
		response.Index = r.log.NextIndex()
		return nil
	}

	// Reject the request if the previous log index matches the last included log index, but the previous log term does
	// not match the last included term.
	if r.lastIncludedIndex == request.PrevLogIndex && r.lastIncludedTerm != request.PrevLogTerm {
		r.logger.Debugf(
			"AppendEntries RPC rejected: reason = previous log entry has conflicting term, logIndex = %d, localTerm = %d, remoteTerm = %d",
			request.PrevLogIndex,
			r.lastIncludedTerm,
			request.PrevLogTerm,
		)
		response.Index = r.lastIncludedIndex
		return nil

	}
	if r.lastIncludedIndex < request.PrevLogIndex {
		prevLogEntry, err := r.log.GetEntry(request.PrevLogIndex)
		if err != nil {
			r.logger.Fatalf("failed to get entry from log: error = %v", err)
		}

		// Reject the request if the log has the previous log entry, but its term does not match.
		if prevLogEntry.Term != request.PrevLogTerm {
			r.logger.Debugf(
				"AppendEntries RPC rejected: reason = previous log entry has conflicting term, index = %d, localTerm = %d, remoteTerm = %d",
				request.PrevLogIndex,
				prevLogEntry.Term,
				request.PrevLogTerm,
			)

			// Find the first index of the conflicting term.
			index := request.PrevLogIndex - 1
			for ; index > r.lastIncludedIndex; index-- {
				entry, err := r.log.GetEntry(index)
				if err != nil {
					r.logger.Fatalf("failed to get entry from log: error = %v", err)
				}
				if entry.Term != prevLogEntry.Term {
					break
				}
			}
			response.Index = index + 1
			return nil
		}
	}

	response.Success = true

	var toAppend []*LogEntry
	for i, entry := range request.Entries {
		if r.log.LastIndex() < entry.Index {
			toAppend = request.Entries[i:]
			break
		}

		existing, err := r.log.GetEntry(entry.Index)
		if err != nil {
			r.logger.Fatalf("failed to get entry from log: error = %v", err)
		}
		if !existing.IsConflict(entry) {
			continue
		}

		r.logger.Warnf("truncating log: index = %d", entry.Index)
		if err := r.log.Truncate(entry.Index); err != nil {
			r.logger.Fatalf("failed to truncate log: %v", err)
		}

		// Fall back to the committed configuration if the current one is
		// truncated. This is necessary since a partitioned leader may have
		// received a membership change request.
		if entry.Index <= r.configuration.Index {
			r.nextConfiguration(r.committedConfiguration)
		}

		toAppend = request.Entries[i:]
		break
	}

	if err := r.log.AppendEntries(toAppend); err != nil {
		r.logger.Fatalf("failed to append entries to log: %v", err)
	}

	if request.LeaderCommit > r.commitIndex {
		r.commitIndex = numeric.Min(request.LeaderCommit, r.log.LastIndex())
		r.applyCond.Broadcast()
	}

	return nil
}

// sendAppendEntriesToPeers sends an AppendEntries RPC to all nodes.
func (r *Raft) sendAppendEntriesToPeers() {
	// Handle the single node cluster case.
	if r.isSingleServerCluster() {
		if r.log.LastIndex() > r.commitIndex {
			r.commitCond.Broadcast()
		}
		r.tryApplyReadOnlyOperations()
	}

	numResponses := 1
	for id, address := range r.configuration.Members {
		if id != r.id {
			go r.sendAppendEntries(id, address, &numResponses)
		}
	}
}

// sendAppendEntries sends an AppendEntries RPC to a node with the provided ID
// and address.
func (r *Raft) sendAppendEntries(id string, address string, numResponses *int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Only leader may send AppendEntries RPCs.
	// It's also possible that this node was removed from the cluster.
	if r.state != Leader || !r.isMember(id) {
		return
	}

	follower := r.followers[id]

	// Send a snapshot instead if the follower log no longer contains the previous log entry.
	if follower.nextIndex <= r.lastIncludedIndex {
		r.sendInstallSnapshot(id, address)
		return
	}

	nextIndex := follower.nextIndex
	prevLogIndex := numeric.Max(nextIndex-1, r.lastIncludedIndex)
	prevLogTerm := r.lastIncludedTerm

	if prevLogIndex > r.lastIncludedIndex && prevLogIndex < r.log.NextIndex() {
		prevEntry, err := r.log.GetEntry(prevLogIndex)
		if err != nil {
			r.logger.Fatalf("failed getting entry from log: error = %v", err)
		}
		prevLogTerm = prevEntry.Term
	}

	entries := make([]*LogEntry, 0, r.log.NextIndex()-nextIndex)
	for index := nextIndex; index > r.lastIncludedIndex && index < r.log.NextIndex(); index++ {
		entry, err := r.log.GetEntry(index)
		if err != nil {
			r.logger.Fatalf("failed getting entry from log: error = %v", err)
		}
		entries = append(entries, entry)
	}

	request := AppendEntriesRequest{
		Term:         r.currentTerm,
		LeaderID:     r.id,
		PrevLogIndex: prevLogIndex,
		PrevLogTerm:  prevLogTerm,
		Entries:      entries,
		LeaderCommit: r.commitIndex,
	}

	r.mu.Unlock()
	response, err := r.transport.SendAppendEntries(address, request)
	r.mu.Lock()

	// Quit if RPC failed, leadership was lost, or the node was removed from the cluster.
	if !r.isMember(id) || err != nil || r.state != Leader {
		return
	}

	// Become a follower if a follower has a more up-to-date term.
	if response.Term > r.currentTerm {
		r.becomeFollower(id, response.Term)
		return
	}

	// If the majority of cluster acknowledges the request, this node is a legitimate leader.
	// Try to apply pending read-only operations.
	if numResponses != nil {
		*numResponses += 1
		if r.hasQuorum(*numResponses) {
			r.tryApplyReadOnlyOperations()
			numResponses = nil
		}
	}

	if !response.Success {
		follower.nextIndex = response.Index

		// Send a snapshot to the follower if the log no longer contains the previous entry.
		if follower.nextIndex <= r.lastIncludedIndex {
			r.sendInstallSnapshot(id, address)
		}

		return
	}

	// Update the next and match index of the followers.
	if request.PrevLogIndex+uint64(len(entries)) > follower.matchIndex {
		follower.nextIndex = numeric.Max(
			follower.nextIndex,
			request.PrevLogIndex+uint64(len(entries))+1,
		)
		follower.matchIndex = request.PrevLogIndex + uint64(len(entries))
		if follower.matchIndex > r.commitIndex {
			r.commitCond.Broadcast()
		}
	}
}

// RequestVote handles vote requests from other nodes during elections. It takes a vote request
// and fills the response with the result of the vote. This will return an error if the node is
// shutdown.
func (r *Raft) RequestVote(request *RequestVoteRequest, response *RequestVoteResponse) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state == Shutdown {
		return fmt.Errorf("could not execute RequestVote RPC: %s is shutdown", r.id)
	}

	r.logger.Debugf(
		"RequestVote RPC received: candidateID = %s, prevote = %t, term = %d, lastLogIndex = %d, lastLogTerm = %d",
		request.CandidateID,
		request.Prevote,
		request.Term,
		request.LastLogIndex,
		request.LastLogTerm,
	)

	response.Term = r.currentTerm
	response.VoteGranted = false

	// This check is necessary to prevent disruptive servers.
	//
	// Ignore any requests for a vote if:
	// 1. This node is a leader and it has recently has successful contact with
	//    a majority of the cluster (it has a valid lease).
	// 2. This node is a follower and it has been recently contacted by the leader.
	if r.operationManager.leaderLease.isValid() ||
		time.Since(r.lastContact) < r.options.electionTimeout {
		r.logger.Debugf(
			"RequestVote RPC rejected: reason = recent contact from leader, knownLeader = %s",
			r.leaderID,
		)
		return nil
	}

	// Reject the request if the term is out-of-date.
	if request.Term < r.currentTerm {
		r.logger.Debugf(
			"RequestVote RPC rejected: reason = out of date term, localTerm =  %d, remoteTerm = %d",
			r.currentTerm,
			request.Term,
		)
		return nil
	}

	// If this is not a prevote and the request has a more up-to-date term,
	// update the current term and become a follower.
	if !request.Prevote && request.Term > r.currentTerm {
		r.becomeFollower(request.CandidateID, request.Term)
		response.Term = r.currentTerm
	}

	// Reject the request if this node has already voted.
	if !request.Prevote && r.votedFor != "" && r.votedFor != request.CandidateID {
		r.logger.Debugf(
			"RequestVote RPC rejected: reason = already voted, votedFor = %s",
			r.votedFor,
		)
		return nil
	}

	// Reject any requests with an out-of-date log.
	//
	// To determine which log is more up-to-date:
	// 1. If the logs have last entries with different terms, then the log with the
	//    greater term is more up-to-date.
	// 2. If the logs end with the same term, the longer log is more up-to-date.
	if request.LastLogTerm < r.log.LastTerm() ||
		(request.LastLogTerm == r.log.LastTerm() && r.log.LastIndex() > request.LastLogIndex) {
		r.logger.Debugf(
			"RequestVote RPC rejected: reason = out-of-date log, localLastLogIndex = %d, localLastLogTerm = %d, remoteLastLogIndex = %d, remoteLastLogTerm = %d",
			r.log.LastIndex(),
			r.log.LastTerm(),
			request.LastLogIndex,
			request.LastLogTerm,
		)
		return nil
	}

	response.VoteGranted = true

	// Only vote if this is a real election.
	if !request.Prevote {
		r.lastContact = time.Now()
		r.votedFor = request.CandidateID
		r.persistTermAndVote()
	}

	r.logger.Infof(
		"RequestVote RPC successful: prevote = %t, votedFor = %s, term = %d",
		request.Prevote,
		request.CandidateID,
		r.currentTerm,
	)

	return nil
}

// electionTicker is a long running loop that will periodically
// send a signal to the election loop to start an election.
func (r *Raft) electionTicker() {
	defer r.wg.Done()

	for {
		// Sleep for a random amount of time between one and two election timeouts
		// to avoid kicking off an election at the same time as other nodes.
		timeout := random.RandomTimeout(r.options.electionTimeout, 2*r.options.electionTimeout)
		time.Sleep(timeout * time.Millisecond)

		r.mu.Lock()
		if r.state == Shutdown {
			r.mu.Unlock()
			return
		}
		r.mu.Unlock()

		r.electionCond.Broadcast()
	}
}

// electionLoop is a long running loop that will attempt to start an
// election when signaled.
func (r *Raft) electionLoop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer r.wg.Done()

	for r.state != Shutdown {
		r.electionCond.Wait()
		r.election()
	}
}

// election will start an election if this node is a voting member
// of the cluster and this node has not been contacted by the leader
// within an election timeout.
func (r *Raft) election() {
	if r.state == Leader || r.state == Shutdown || !r.isVoter(r.id) ||
		time.Since(r.lastContact) < r.options.electionTimeout {
		return
	}
	if r.state == Follower {
		r.becomePreCandidate()
	}
	if r.state == Candidate {
		r.becomeCandidate()
	}

	r.sendRequestVoteToPeers()
}

// sendRequestVoteToPeers sends a RequestVoteRPC to all nodes in the cluster,
// excluding those that are non-voters.
func (r *Raft) sendRequestVoteToPeers() {
	// Handle the single node cluster case.
	if r.isSingleServerCluster() {
		r.becomeLeader()
		return
	}

	// Send RequestVote RPCs to all voting members of the cluster.
	votesRecieved := 1
	isPrevote := r.state == PreCandidate
	for id, address := range r.configuration.Members {
		if id != r.id && r.isVoter(id) {
			go r.sendRequestVote(id, address, &votesRecieved, isPrevote)
		}
	}
}

// sendRequestVote sends a RequestVote RPC to the node with the provided
// ID and address if it is a voting member.
func (r *Raft) sendRequestVote(id string, address string, votes *int, prevote bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Do not send requests to non-voting members and only send
	// requests if this node is a voting member of the cluster.
	if !r.isVoter(id) || !r.isVoter(r.id) {
		return
	}

	request := RequestVoteRequest{
		CandidateID:  r.id,
		Term:         r.currentTerm,
		LastLogIndex: r.log.LastIndex(),
		LastLogTerm:  r.log.LastTerm(),
		Prevote:      prevote,
	}

	// Use the term that would be used in the election if this is a prevote.
	if prevote {
		request.Term++
	}

	r.mu.Unlock()
	response, err := r.transport.SendRequestVote(address, request)
	r.mu.Lock()

	if err != nil || r.state == Shutdown {
		return
	}

	// Ensure this response is not stale. It is possible that this node has started another election.
	if r.currentTerm > request.Term {
		return
	}

	// Increment vote count if vote is granted.
	if response.VoteGranted {
		*votes++
	}

	// Become a follower if a recipient has a more up-to-date term.
	if response.Term > request.Term {
		r.becomeFollower(id, response.Term)
		return
	}

	// If this is a prevote and a majority of the cluster respond with success to this node's
	// vote requests, become a candidate.
	if r.hasQuorum(*votes) && r.state == PreCandidate {
		// Signal to the election loop to start an election so that the real election
		// does not have to wait until the election ticker goes off again.
		r.state = Candidate
		r.electionCond.Broadcast()
	}

	// If this an election and a majority of the cluster vote for this node, become the leader.
	if !prevote && r.hasQuorum(*votes) && r.state == Candidate {
		r.becomeLeader()
	}
}

// InstallSnapshot handles snapshot installation requests from the leader. It takes a request to
// install a snapshot and fills the response with the result of the installation. This will
// return an error if the node is shutdown.
func (r *Raft) InstallSnapshot(
	request *InstallSnapshotRequest,
	response *InstallSnapshotResponse,
) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state == Shutdown {
		return fmt.Errorf("could not execute InstallSnapshot RPC: %s is shutdown", r.id)
	}

	r.logger.Debugf(
		"InstallSnapshot RPC received: leaderID = %s, term = %d, lastIndex = %d, lastTerm = %d, offset = %d, done = %v",
		request.LeaderID,
		request.Term,
		request.LastIncludedIndex,
		request.LastIncludedTerm,
		request.Offset,
		request.Done,
	)

	response.Term = r.currentTerm

	// Reject the request if the term is out-of-date.
	if r.currentTerm > request.Term {
		r.logger.Debugf(
			"InstallSnapshot RPC rejected: reason = out of date term, localTerm = %d, remoteTerm = %d",
			r.currentTerm,
			request.Term,
		)
		return nil
	}

	// If the request has a more up-to-date term, update current term and become a follower.
	if r.currentTerm < request.Term {
		r.becomeFollower(request.LeaderID, request.Term)
		response.Term = request.Term
	}

	// Transition to follower state if still in candidate or precandidate state.
	if request.Term == r.currentTerm && (r.state == Candidate || r.state == PreCandidate) {
		r.becomeFollower(request.LeaderID, request.Term)
	}

	r.lastContact = time.Now()

	// The received snapshot does not contain anything new.
	if r.lastIncludedIndex >= request.LastIncludedIndex ||
		r.lastApplied >= request.LastIncludedIndex {
		return nil
	}

	// Discard an incomplete snapshot if this request has a greater last index.
	if r.snapshot != nil {
		metadata := r.snapshot.Metadata()
		if metadata.LastIncludedIndex < request.LastIncludedIndex {
			if err := r.snapshot.Discard(); err != nil {
				r.logger.Fatalf("failed to discard snapshot: error = %v", err)
			}
			r.snapshot = nil
		}
	}

	// Create a new snapshot file if one has not already been created.
	if r.snapshot == nil {
		snapshot, err := r.snapshotStorage.NewSnapshotFile(
			request.LastIncludedIndex,
			request.LastIncludedTerm,
			request.Configuration,
		)
		if err != nil {
			r.logger.Fatalf("failed to create snapshot file: error = %v", err)
		}
		r.snapshot = snapshot
	}

	// Ensure the offset in the request is the expected offset.
	offset, err := r.snapshot.Seek(0, io.SeekCurrent)
	response.BytesWritten = offset
	if err != nil {
		r.logger.Fatalf("failed to seek snapshot file: error = %v", err)
	}
	if request.Offset != offset {
		r.logger.Warnf(
			"InstallSnapshot RPC contains incorrect offset: expectedOffset = %d, receivedOffset = %d",
			offset,
			request.Offset,
		)
		return nil
	}

	// Write the snapshot chunk to the file.
	reader := bytes.NewReader(request.Bytes)
	n, err := io.Copy(r.snapshot, reader)
	if err != nil {
		r.logger.Fatalf("failed to write snapshot file: error = %v", err)
	}
	response.BytesWritten += n

	if !request.Done {
		return nil
	}

	if err := r.snapshot.Close(); err != nil {
		r.logger.Fatalf("failed to close snapshot file: error = %v", err)
	}

	r.snapshot = nil
	r.lastIncludedIndex = request.LastIncludedIndex
	r.lastIncludedTerm = request.LastIncludedTerm

	// If an existing log entry has the same index and term as the last index
	// and last term, discard the log through the last index and reply.
	if entry, _ := r.log.GetEntry(request.LastIncludedIndex); entry != nil &&
		entry.Term == request.LastIncludedTerm {
		// Wait for all operations up to last included index have been applied before compacting the log.
		for r.state != Shutdown && r.lastApplied < request.LastIncludedIndex {
			r.applyCond.Wait()
		}

		// It's possible that a snapshot was taken and the log was compacted while the lock was released.
		if r.state == Shutdown || r.lastIncludedIndex > request.LastIncludedIndex {
			return nil
		}

		r.logger.Warnf("compacting log: logIndex = %d", request.LastIncludedIndex)
		if err := r.log.Compact(request.LastIncludedIndex); err != nil {
			r.logger.Fatalf("failed to compact log: error = %v", err)
		}

		return nil
	}

	snapshot, err := r.snapshotStorage.SnapshotFile()
	if err != nil {
		r.logger.Fatalf("failed to get snapshot file: error = %v", err)
	}

	// Restore the state machine with the snapshot.
	// This could take a while so it's probably best that the lock is released.
	r.mu.Unlock()
	r.logger.Warnf(
		"restoring state machine with snapshot: lastIndex = %d, lastTerm = %d",
		request.LastIncludedIndex,
		request.LastIncludedTerm,
	)
	if err := r.fsm.Restore(snapshot); err != nil {
		r.logger.Fatalf("failed to restore state machine with snapshot: error = %v", err)
	}
	if err := snapshot.Close(); err != nil {
		r.logger.Fatalf("failed to close snapshot file: error = %v", err)
	}
	r.mu.Lock()

	if r.state == Shutdown {
		return nil
	}

	r.lastApplied = request.LastIncludedIndex
	r.commitIndex = request.LastIncludedIndex

	// Discard the entire log.
	r.logger.Warnf(
		"discarding log: lastIndex = %d, lastTerm = %d",
		request.LastIncludedIndex,
		request.LastIncludedTerm,
	)
	if err := r.log.DiscardEntries(request.LastIncludedIndex, request.LastIncludedTerm); err != nil {
		r.logger.Fatalf("failed to discard log entries: error = %v", err)
	}

	// Update the configuration.
	r.applyConfiguration(request.Configuration)

	r.logger.Infof(
		"snapshot installation completed successfully: lastIndex = %d, lastTerm = %d",
		request.LastIncludedIndex,
		request.LastIncludedTerm,
	)

	return nil
}

// snapshotLoop is a long running loop that will takes a snapshot of the
// state machine when signaled if one is necessary.
func (r *Raft) snapshotLoop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer r.wg.Done()

	for r.state != Shutdown {
		r.snapshotCond.Wait()
		if r.fsm.NeedSnapshot(r.log.Size()) {
			r.takeSnapshot()
		}
	}
}

// takeSnapshot takes a snapshot of the state machine. A snapshot will
// only be taken if there is new state since the previous snapshot and there
// is not a pending configuration change.
func (r *Raft) takeSnapshot() {
	// There is nothing new to snapshot.
	if r.lastApplied <= r.lastIncludedIndex {
		return
	}

	// Put off snapshots if there is an outstanding configuration change.
	if r.committedConfiguration == nil || r.committedConfiguration.Index > r.lastApplied {
		return
	}

	lastAppliedEntry, err := r.log.GetEntry(r.lastApplied)
	if err != nil {
		r.logger.Fatalf("failed to get entry from log: error = %v", err)
	}

	r.logger.Infof(
		"starting to take snapshot: lastIndex = %d, lastTerm = %d",
		lastAppliedEntry.Index,
		lastAppliedEntry.Term,
	)

	// Create a new snapshot file.
	configurationData := r.encodeConfiguration(r.committedConfiguration)
	snapshot, err := r.snapshotStorage.NewSnapshotFile(
		lastAppliedEntry.Index,
		lastAppliedEntry.Term,
		configurationData,
	)
	if err != nil {
		r.logger.Fatalf("failed to create snapshot file: error = %v", err)
	}

	// Take a snapshot of the state machine.
	// It's best that the lock is not held here since this might take a while.
	r.mu.Unlock()
	if err := r.fsm.Snapshot(snapshot); err != nil {
		r.logger.Fatalf("failed to take snapshot of state machine: error = %v", err)
	}
	if err := snapshot.Close(); err != nil {
		r.logger.Fatalf("failed to close snapshot file: error = %v", err)
	}
	r.mu.Lock()

	// It's possible a snapshot was installed and the log was compacted while the lock was released.
	if lastAppliedEntry.Index <= r.lastIncludedIndex {
		return
	}

	// Compact the log.
	r.lastIncludedIndex = lastAppliedEntry.Index
	r.lastIncludedTerm = lastAppliedEntry.Term
	r.logger.Warnf("compacting log: logIndex = %d", r.lastIncludedIndex)
	if err := r.log.Compact(r.lastIncludedIndex); err != nil {
		r.logger.Fatalf("failed to compact log: error = %v", err)
	}
	r.resetSnapshotFiles()

	r.logger.Infof(
		"snapshot taken successfully: lastIndex = %d, lastTerm = %d",
		r.lastIncludedIndex,
		r.lastIncludedTerm,
	)
}

// sendInstallSnapshot sends a snapshot to the node with the provided ID and address.
func (r *Raft) sendInstallSnapshot(id, address string) {
	// Only the leader may send snapshots.
	if r.state != Leader {
		return
	}

	if r.lastIncludedIndex == 0 {
		r.logger.Warn("cannot send snapshot to follower because one has not been taken")
		return
	}

	follower := r.followers[id]

	// Retrieve the most recent snapshot file to send to the follower if one is not already open.
	if follower.snapshot == nil {
		snapshot, err := r.snapshotStorage.SnapshotFile()
		if err != nil {
			r.logger.Fatalf("failed to get snapshot file: error = %v", err)
		}
		follower.snapshot = snapshot
	}

	metadata := follower.snapshot.Metadata()
	offset, err := follower.snapshot.Seek(0, io.SeekCurrent)
	if err != nil {
		r.logger.Fatalf("failed to seek snapshot file: error = %v", err)
	}

	request := InstallSnapshotRequest{
		LeaderID:          r.id,
		Term:              r.currentTerm,
		LastIncludedIndex: metadata.LastIncludedIndex,
		LastIncludedTerm:  metadata.LastIncludedTerm,
		Configuration:     metadata.Configuration,
		Offset:            offset,
	}

	// Read a chunk of the snapshot from the file.
	var buf bytes.Buffer
	n, err := io.Copy(&buf, follower.snapshot)
	if err != nil {
		if err := follower.snapshot.Close(); err != nil {
			r.logger.Errorf("failed to close snapshot file: error = %v", err)
		}
		r.logger.Fatalf("failed to read snapshot file: error = %v", err)
	}
	request.Bytes = buf.Bytes()
	request.Done = n < snapshotChunkSize

	r.mu.Unlock()
	response, err := r.transport.SendInstallSnapshot(address, request)
	r.mu.Lock()

	if follower.snapshot == nil || err != nil {
		return
	}

	// If the follower has a more up-to-date term, transition to the follower state.
	if response.Term > r.currentTerm {
		r.becomeFollower(id, response.Term)
		return
	}

	// The follower is either missing part of the snapshot or already has this part.
	// Reset to the follower's offset.
	if response.BytesWritten != offset {
		if _, err := follower.snapshot.Seek(response.BytesWritten, io.SeekStart); err != nil {
			r.logger.Fatalf("failed to seek snapshot file: error = %v", err)
		}
		return
	}

	if !request.Done {
		return
	}

	if err := follower.snapshot.Close(); err != nil {
		r.logger.Fatalf("failed to close snapshot file: error = %v", err)
	}
	follower.snapshot = nil
	follower.matchIndex = request.LastIncludedIndex
	follower.nextIndex = request.LastIncludedIndex + 1
}

// heartbeatLoop is a long running loop that periodically sends heartbeats to
// followers if this node is the leader.
func (r *Raft) heartbeatLoop() {
	defer r.wg.Done()

	// If this node is the leader, broadcast heartbeat messages to followers
	// once every heartbeat interval.
	for {
		time.Sleep(r.options.heartbeatInterval)
		r.mu.Lock()
		if r.state == Shutdown {
			r.mu.Unlock()
			return
		}
		if r.state == Follower {
			r.mu.Unlock()
			continue
		}
		r.sendAppendEntriesToPeers()
		r.mu.Unlock()
	}
}

// commitLoop is a long running loop that verifies quorum has been achieved
// for log entries and updates the commit index of the leader.
func (r *Raft) commitLoop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer r.wg.Done()

	for r.state != Shutdown {
		r.commitCond.Wait()

		// Followers may not commit log entries.
		if r.state != Leader {
			continue
		}

		committed := false

		for index := r.commitIndex + 1; index <= r.log.LastIndex(); index++ {
			// It is not safe for the leader to commit an entry with a term
			// different from the current term. It is possible for a log entry
			// to be agreed upon by the majority of node in the cluster, but
			// be overwritten by a future leader.
			if entry, err := r.log.GetEntry(index); err != nil {
				r.logger.Fatalf("failed to get  entry from log: error = %v", err)
			} else if entry.Term != r.currentTerm {
				continue
			}

			// Check whether the majority of nodes in the cluster agree on the entry.
			// If they do, it is safe to commit.
			matches := 1
			for id, follower := range r.followers {
				// Ignore this node and any nodes which are not voting members.
				if id == r.id || !r.configuration.IsVoter[id] {
					continue
				}
				if follower.matchIndex >= index {
					matches++
				}
			}

			if r.hasQuorum(matches) {
				r.commitIndex = index
				committed = true
			}
		}

		if committed {
			r.applyCond.Broadcast()
			r.sendAppendEntriesToPeers()
		}
	}
}

// applyLoop is a long running loop that applies replicated operations to the state machine.
func (r *Raft) applyLoop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer r.wg.Done()

	for r.state != Shutdown {
		r.applyCond.Wait()

		// Scan the log starting at the entry following the last applied entry
		// and apply any entries that have been committed.
		for r.lastApplied < r.commitIndex && r.state != Shutdown {
			entry, err := r.log.GetEntry(r.lastApplied + 1)
			if err != nil {
				r.logger.Fatalf("failed to get entry from log: error = %v", err)
			}

			switch entry.EntryType {
			case NoOpEntry:
			case ConfigurationEntry:
				r.applyConfiguration(entry.Data)
				respond(r.configurationResponseCh, *r.configuration, nil)
			case OperationEntry:
				responseCh := r.operationManager.pendingReplicated[entry.Index]
				delete(r.operationManager.pendingReplicated, entry.Index)

				operation := Operation{
					LogIndex:      entry.Index,
					LogTerm:       entry.Term,
					Bytes:         entry.Data,
					OperationType: Replicated,
				}
				lastApplied := r.lastApplied

				r.mu.Unlock()
				response := OperationResponse{
					Operation:           operation,
					ApplicationResponse: r.fsm.Apply(&operation),
				}
				respond(responseCh, response, nil)
				r.logger.Debugf(
					"applied operation to state machine: logIndex = %d, logTerm = %d, type = %s",
					operation.LogIndex,
					operation.LogTerm,
					operation.OperationType.String(),
				)
				r.mu.Lock()

				// It's possible a snapshot was installed while the lock was released.
				// It's not safe to increment the last applied index if it has changed.
				if r.lastApplied != lastApplied {
					continue
				}
			default:
				r.logger.Fatal("log entry has invalid type")
			}

			r.lastApplied++
			if r.fsm.NeedSnapshot(r.log.Size()) {
				r.snapshotCond.Signal()
			}
		}

		if r.state == Leader {
			r.readOnlyCond.Broadcast()
		}
	}
}

// applyConfigurationEntry applies a log entry containing a configuration to this node.
func (r *Raft) applyConfiguration(configurationData []byte) {
	configuration := r.decodeConfiguration(configurationData)
	if r.committedConfiguration != nil && configuration.Index <= r.committedConfiguration.Index {
		return
	}
	r.nextConfiguration(&configuration)
	r.committedConfiguration = &configuration
}

// readOnlyLoop is a long running loop that applies read-only operations to the state machine.
func (r *Raft) readOnlyLoop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer r.wg.Done()

	for r.state != Shutdown {
		r.readOnlyCond.Wait()
		// Only the leader may apply read-only operations and it is
		// only safe to apply them once the leader has committed
		// at least one log entry.
		if r.state != Leader || !r.committedThisTerm() {
			continue
		}

		appliableOperations := r.operationManager.appliableReadOnlyOperations(r.lastApplied)
		for operation, responseCh := range appliableOperations {
			if operation.OperationType == LeaseBasedReadOnly &&
				!r.operationManager.leaderLease.isValid() {
				respond(responseCh, OperationResponse{}, ErrInvalidLease)
				continue
			}

			r.mu.Unlock()
			response := OperationResponse{
				Operation:           *operation,
				ApplicationResponse: r.fsm.Apply(operation),
			}
			respond(responseCh, response, nil)
			r.logger.Debugf(
				"applied operation to state machine: readIndex = %d, type = %s",
				operation.readIndex,
				operation.OperationType.String(),
			)
			r.mu.Lock()

			if r.state != Leader {
				break
			}
		}
	}
}

// becomeCandidate transitions this node to the candidate state.
// This will increment the term and cast a vote for this node.
func (r *Raft) becomeCandidate() {
	r.state = Candidate
	r.currentTerm++
	r.votedFor = r.id
	r.persistTermAndVote()
	r.logger.Infof("entered the candidate state: term = %d", r.currentTerm)
}

// becomePreCandidate transitions this node to the precandidate state.
// This does not increment the term or cast a vote for this node.
func (r *Raft) becomePreCandidate() {
	r.state = PreCandidate
	r.logger.Infof("entered the pre-candidate state: term = %d", r.currentTerm)
}

// becomeLeader transitions this node to the leader state.
func (r *Raft) becomeLeader() {
	r.state = Leader
	r.operationManager = newOperationManager(r.options.leaseDuration)
	for _, follower := range r.followers {
		follower.nextIndex = r.log.LastIndex() + 1
		follower.matchIndex = 0
	}
	r.resetSnapshotFiles()

	// Append a new log entry for this term.
	entry := NewLogEntry(r.log.NextIndex(), r.currentTerm, []byte{}, NoOpEntry)
	if err := r.log.AppendEntry(entry); err != nil {
		r.logger.Fatal("failed to append entry to log: error = %v", err)
	}
	r.sendAppendEntriesToPeers()

	r.logger.Infof("entered the leader state: term = %d", r.currentTerm)
}

// becomeFollower transitions this node to the follower state.
func (r *Raft) becomeFollower(leaderID string, term uint64) {
	r.state = Follower
	r.currentTerm = term
	r.leaderID = leaderID
	r.votedFor = ""
	r.persistTermAndVote()
	r.resetSnapshotFiles()

	// Cancel any pending operations.
	r.operationManager.notifyLostLeaderShip(r.id, r.leaderID)
	r.operationManager = newOperationManager(r.options.leaseDuration)

	r.logger.Infof("entered the follower state: term = %d", r.currentTerm)
}

// stepDown transitions a node from the leader state to the follower state when it
// has been removed from the cluster. Unlike becomeFollower, stepDown does not persist
// the current term and vote.
func (r *Raft) stepdown() {
	r.state = Follower

	// Cancel any pending operations.
	r.operationManager.notifyLostLeaderShip(r.id, r.leaderID)
	r.operationManager = newOperationManager(r.options.leaseDuration)

	r.logger.Info("stepped down to the follower state")
}

// tryApplyReadOnlyOperations renews the lease and notifies the read-only
// loop that it may be possible to apply some read-only operations.
func (r *Raft) tryApplyReadOnlyOperations() {
	r.operationManager.markAsVerified()
	r.operationManager.leaderLease.renew()
	r.operationManager.shouldVerifyQuorum = true
	r.readOnlyCond.Broadcast()
}

// resetSnapshotFiles closes or discards any open snapshot files on this node.
// If the file is being written, it is discarded. If it is being read, it is closed.
func (r *Raft) resetSnapshotFiles() {
	for _, follower := range r.followers {
		if follower.snapshot != nil {
			if err := follower.snapshot.Close(); err != nil {
				r.logger.Fatalf("failed to close snapshot file: error = %v", err)
			}
			follower.snapshot = nil
		}
	}
	if r.snapshot != nil {
		if err := r.snapshot.Discard(); err != nil {
			r.logger.Fatalf("failed to discard snapshot file: error = %v", err)
		}
		r.snapshot = nil
	}
}

// hasQuorum returns true if the provided count makes constitutes
// quorum for the cluster and false otherwise. Non-voting members
// of the cluster are not considered for quorum.
func (r *Raft) hasQuorum(count int) bool {
	voters := 0
	for _, isVoter := range r.configuration.IsVoter {
		if isVoter {
			voters++
		}
	}
	return count > voters/2
}

// committedThisTerm returns true if a log entry from the current term
// has been committed and false otherwise.
func (r *Raft) committedThisTerm() bool {
	// The log contains the commit index.
	// Check if the term of the committed entry  matches the current term.
	if r.log.Contains(r.commitIndex) {
		entry, err := r.log.GetEntry(r.commitIndex)
		if err != nil {
			r.logger.Fatalf("failed to get entry from log: error = %v", err)
		}
		return entry.Term == r.currentTerm
	}

	// The log must have been compacted.
	// Check if the snapshot contains a log entry from this term.
	return r.lastIncludedTerm == r.currentTerm
}

// persistTermAndVote writes the term and vote for this node to non-volatile storage.
func (r *Raft) persistTermAndVote() {
	if err := r.stateStorage.SetState(r.currentTerm, r.votedFor); err != nil {
		r.logger.Fatalf("failed to persist term and vote: error = %v", err)
	}
}

// nextConfiguration transitions this node from its current configuration to
// to the provided configuration. Any unused connections will be torn down
// and any new connections will be established. If this node is not a member
// of the next configuration, it will transition to a configuration consisting
// of only itself with non-voter status.
func (r *Raft) nextConfiguration(next *Configuration) {
	r.logger.Infof("transitioning to new configuration: configuration = %s", next.String())

	defer func() {
		r.configuration = next
	}()

	// Step down if this node is being removed and it is the leader.
	if _, ok := next.Members[r.id]; !ok {
		if r.state == Leader {
			r.stepdown()
		}
		r.resetSnapshotFiles()
	}

	// Delete removed nodes from followers.
	for id := range r.configuration.Members {
		if _, ok := next.Members[id]; !ok {
			delete(r.followers, id)
		}
	}

	// Create entry for added nodes.
	for id := range next.Members {
		if _, ok := r.configuration.Members[id]; !ok {
			r.followers[id] = new(follower)
		}
	}
}

// encodeConfiguration encodes the provided configuration.
func (r *Raft) encodeConfiguration(configuration *Configuration) []byte {
	data, err := r.transport.EncodeConfiguration(configuration)
	if err != nil {
		r.logger.Fatalf("failed to encode configuration: error = %v", err)
	}
	return data
}

// decodeConfiguration decodes the provided bytes into a configuration.
func (r *Raft) decodeConfiguration(data []byte) Configuration {
	configuration, err := r.transport.DecodeConfiguration(data)
	if err != nil {
		r.logger.Fatalf("failed to decode configuration: error = %v", err)
	}
	return configuration
}

// appendConfiguration sets the log index associated with the
// configuration and appends it to the log.
func (r *Raft) appendConfiguration(configuration *Configuration) {
	configuration.Index = r.log.NextIndex()
	data := r.encodeConfiguration(configuration)
	entry := NewLogEntry(configuration.Index, r.currentTerm, data, ConfigurationEntry)
	if err := r.log.AppendEntry(entry); err != nil {
		r.logger.Fatalf("failed to append entry to log: error = %v", err)
	}
}

// isVoter returns true if the node with the provided ID
// is a voting member of the cluster and false otherwise.
func (r *Raft) isVoter(id string) bool {
	return r.configuration.IsVoter[id]
}

// isMember returns true if the node with the provided ID
// is a member of the cluster and false otherwise.
func (r *Raft) isMember(id string) bool {
	_, ok := r.configuration.Members[id]
	return ok
}

// isSingleServerCluster returns true if the current configuration only contains
// this node as a voting member.
func (r *Raft) isSingleServerCluster() bool {
	return len(r.configuration.Members) == 1 && r.configuration.IsVoter[r.id]
}

// pendingConfigurationChange returns true if the current configuration
// has not been committed.
func (r *Raft) pendingConfigurationChange() bool {
	return r.committedConfiguration == nil ||
		r.committedConfiguration.Index != r.configuration.Index
}
