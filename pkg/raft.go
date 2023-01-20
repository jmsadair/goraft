package raft

import (
	"sync"
	"time"

	"github.com/jmsadair/raft/internal/errors"
	logger "github.com/jmsadair/raft/internal/logger"
	pb "github.com/jmsadair/raft/internal/protobuf"
	"github.com/jmsadair/raft/internal/util"
)

type Raft struct {
	id           string
	options      options
	peers        []*Peer
	log          Log
	storage      Storage
	state        State
	commitIndex  uint64
	lastApplied  uint64
	currentTerm  uint64
	votedFor     string
	lastContact  time.Time
	submissionCh chan interface{}
	shutdownCh   chan interface{}
	mu           sync.Mutex
}

func newRaft(id string, peers []*Peer, log Log, storage Storage, opts ...Option) (*Raft, error) {
	var options options
	for _, opt := range opts {
		if err := opt(&options); err != nil {
			return nil, errors.WrapError(err, "failed to create new raft: %s", err.Error())
		}
	}

	if options.logger == nil {
		logger, err := logger.NewLogger()
		if err != nil {
			return nil, errors.WrapError(err, "failed to create new raft: %s", err.Error())
		}
		options.logger = logger
	}

	if options.heartbeatInterval == 0 {
		options.heartbeatInterval = defaultHeartbeat
	}

	if options.electionTimeout == 0 {
		options.electionTimeout = defaultElectionTimeout
	}

	currentTermKey := []byte("currentTerm")
	currentTerm, err := storage.GetUint64(currentTermKey)
	if err != nil {
		return nil, errors.WrapError(err, "failed to restore current term from storage: %s", err.Error())
	}

	votedForKey := []byte("votedFor")
	votedForBytes, err := storage.Get(votedForKey)
	if err != nil {
		return nil, errors.WrapError(err, "failed to restore vote from storage: %s", err.Error())
	}
	votedFor := string(votedForBytes)

	raft := &Raft{
		id:          id,
		options:     options,
		peers:       peers,
		log:         log,
		storage:     storage,
		state:       Shutdown,
		currentTerm: currentTerm,
		votedFor:    votedFor,
	}

	return raft, nil
}

func (r *Raft) start() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state != Shutdown {
		return
	}

	for _, peer := range r.peers {
		if err := peer.connect(); err != nil {
			r.options.logger.Errorf("error connecting to peer: %s", err.Error())
		}
	}

	r.shutdownCh = make(chan interface{})
	r.lastContact = time.Now()
	r.becomeCandidate()
	go r.mainLoop()

	r.options.logger.Infof("raft server with ID %s started", r.id)
}

func (r *Raft) stop() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.state == Shutdown {
		return
	}

	r.state = Shutdown
	close(r.shutdownCh)
	r.options.logger.Infof("raft server with ID %s stopped", r.id)
}

func (r *Raft) replicate(command []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.state != Leader {
		return errors.WrapError(nil, "%s is not the leader", r.id)
	}
	entry := NewLogEntry(r.log.LastIndex(), r.currentTerm, command)
	r.log.AppendEntry(entry)
	r.submissionCh <- struct{}{}
	return nil
}

func (r *Raft) appendEntries(request *pb.AppendEntriesRequest) *pb.AppendEntriesResponse {
	r.mu.Lock()
	defer r.mu.Unlock()

	var prevLogEntry *LogEntry
	var err error

	response := &pb.AppendEntriesResponse{
		Term:    r.currentTerm,
		Success: false,
	}

	if request.GetTerm() < r.currentTerm {
		r.options.logger.Debugf("rejecting request to append entries: out of date term: term = %d, request term = %d",
			r.currentTerm, request.GetTerm())
		return response
	}

	if request.GetTerm() > r.currentTerm {
		r.becomeFollower(request.GetTerm())
		response.Term = r.currentTerm
	}

	if request.GetPrevLogIndex() != 0 {
		if prevLogEntry, err = r.log.GetEntry(request.GetPrevLogIndex()); err != nil {
			r.options.logger.Debugf("rejecting request to append entries: previous log entry at index %d does not exist: %s",
				request.GetPrevLogIndex(), err.Error())
			return response
		}

		if prevLogEntry.Term() != request.GetPrevLogTerm() {
			r.options.logger.Debugf("rejecting request to append entries: previous log term does not match: local = %d, remote = %d",
				prevLogEntry.Term(), request.GetPrevLogTerm())
			return response
		}
	}

	var toAppend []*LogEntry
	entries := make([]*LogEntry, len(request.GetEntries()))

	for i, entry := range request.GetEntries() {
		entries[i] = &LogEntry{entry: entry}
	}

	for i, entry := range entries {
		if r.log.LastIndex() < entry.Index() {
			toAppend = entries[i:]
			break
		}
		existing, _ := r.log.GetEntry(entry.Index())
		if !existing.IsConflict(entry) {
			continue
		}
		if err = r.log.Truncate(entry.Index()); err != nil {
			r.options.logger.Fatalf("error truncating log: %s", err.Error())
		}
		toAppend = entries[i:]
		break
	}

	if err = r.log.AppendEntries(toAppend); err != nil {
		r.options.logger.Fatalf("error appending entries to log: %s", err.Error())
	}

	if request.GetLeaderCommit() > r.commitIndex {
		r.commitIndex = util.Min(request.GetLeaderCommit(), r.log.LastIndex())
	}

	response.Success = true

	r.lastContact = time.Now()

	return response
}

func (r *Raft) sendAppendEntries() {
	for _, peer := range r.peers {
		go func(peer *Peer) {
			r.mu.Lock()
			defer r.mu.Unlock()

			entries := make([]*pb.LogEntry, 0)
			if r.log.LastIndex() > peer.getNextIndex() {
				entries = make([]*pb.LogEntry, r.log.LastIndex()-peer.getNextIndex())
			}
			for index := peer.getNextIndex(); index <= r.log.LastIndex(); index++ {
				entry, err := r.log.GetEntry(index)
				if err != nil {
					r.options.logger.Fatalf("error getting entry from log: %s", err.Error())
				}
				entries[index-peer.getNextIndex()] = entry.Entry()
			}

			request := &pb.AppendEntriesRequest{
				Term:         r.currentTerm,
				LeaderId:     r.id,
				PrevLogIndex: r.log.LastIndex(),
				PrevLogTerm:  r.log.LastTerm(),
				Entries:      entries,
				LeaderCommit: r.commitIndex,
			}

			response, err := peer.appendEntries(request)
			if err != nil {
				r.options.logger.Errorf("error appending entries to peer: %s", err.Error())
				return
			}

			if response.GetTerm() > r.currentTerm {
				r.becomeFollower(request.GetTerm())
				response.Term = r.currentTerm
				return
			}

			if !response.GetSuccess() {
				peer.setNextIndex(peer.getNextIndex() - 1)
				return
			}

			peer.setNextIndex(peer.getNextIndex() + uint64(len(entries)))
			peer.setMatchIndex(peer.getNextIndex() + uint64(len(entries)) - 1)

			for index := r.commitIndex; index < r.log.LastIndex(); index++ {
				if entry, _ := r.log.GetEntry(index); entry == nil || entry.Term() != r.currentTerm {
					break
				}
				matches := 1
				for _, peer := range r.peers {
					if peer.getMatchIndex() >= index {
						matches += 1
					}
				}
				if matches >= r.quorum() {
					r.commitIndex = index
				}
			}
		}(peer)
	}
}

func (r *Raft) requestVote(request *pb.RequestVoteRequest) *pb.RequestVoteResponse {
	r.mu.Lock()
	defer r.mu.Unlock()

	response := &pb.RequestVoteResponse{
		Term:        r.currentTerm,
		VoteGranted: false,
	}

	if request.GetTerm() < r.currentTerm {
		r.options.logger.Debugf("rejected vote request: out of date term: term = %d, candidate term = %d",
			r.currentTerm, request.GetTerm())
		return response
	}

	if request.GetTerm() > r.currentTerm {
		r.becomeFollower(request.GetTerm())
		response.Term = r.currentTerm
	}

	if r.votedFor != "" && r.votedFor != request.GetCandidateId() {
		r.options.logger.Debugf("rejected vote request: already voted: votedFor = %s, term = %d",
			r.votedFor, r.currentTerm)
		return response
	}

	if request.GetTerm() < r.log.LastTerm() || (request.GetLastLogTerm() == r.log.LastTerm() && r.log.LastIndex() > request.GetLastLogIndex()) {
		r.options.logger.Debugf("rejecting vote request: out of date log: lastIndex = %d, lastTerm = %d, candidate lastIndex = %d, candidate lastTerm = %d",
			r.log.LastIndex(), r.log.LastTerm(), request.GetLastLogIndex(), request.GetLastLogTerm())
		return response
	}

	r.setVotedFor(request.GetCandidateId())
	response.VoteGranted = true
	r.options.logger.Debugf("request for vote granted: %s voted for %s, term = %d", r.id, request.GetCandidateId(), r.currentTerm)

	return response
}

func (r *Raft) leaderLoop() {
	heartbeatInterval := r.options.heartbeatInterval
	heartbeat := time.NewTicker(heartbeatInterval)

	for {
		select {
		case <-heartbeat.C:
			r.sendAppendEntries()
			heartbeat.Reset(heartbeatInterval)
		case <-r.submissionCh:
			r.sendAppendEntries()
		case <-r.shutdownCh:
			return
		default:
			<-time.After(5 * time.Millisecond)
			r.mu.Lock()
			if r.state != Leader {
				r.mu.Unlock()
				return
			}
			r.mu.Unlock()
		}
	}
}

func (r *Raft) followerLoop() {
	electionTimeout := r.options.electionTimeout
	electionTimer := util.RandomTimeout(electionTimeout, electionTimeout*2)

	for {
		select {
		case <-electionTimer:
			electionTimer = util.RandomTimeout(electionTimeout, electionTimeout*2)
			r.mu.Lock()
			if time.Since(r.lastContact) > electionTimeout {
				r.becomeCandidate()
				r.mu.Unlock()
				return
			}
			r.mu.Unlock()
		case <-r.shutdownCh:
			return
		}
	}
}

func (r *Raft) candidateLoop() {
	votesReceived := 1

	r.mu.Lock()
	r.votedFor = r.id
	r.currentTerm++
	r.mu.Unlock()

	responses := make(chan *pb.RequestVoteResponse)

	electionTimeout := r.options.electionTimeout
	electionTimer := util.RandomTimeout(electionTimeout, electionTimeout*2)

	for _, peer := range r.peers {
		go func(peer *Peer) {
			r.mu.Lock()
			request := &pb.RequestVoteRequest{
				CandidateId:  r.id,
				Term:         r.currentTerm,
				LastLogIndex: r.log.LastIndex(),
				LastLogTerm:  r.log.LastTerm(),
			}
			r.mu.Unlock()

			response, err := peer.requestVote(request)
			if err != nil {
				r.options.logger.Errorf("error requesting vote from peer %s: %s", peer.id, err.Error())
				return
			}

			responses <- response
		}(peer)
	}

	for {
		select {
		case response := <-responses:
			if response.VoteGranted {
				votesReceived++
			}
			r.mu.Lock()
			if response.GetTerm() > r.currentTerm {
				r.becomeFollower(response.GetTerm())
				r.mu.Unlock()
				return
			}
			if votesReceived >= r.quorum() {
				r.becomeLeader()
				r.mu.Unlock()
				return
			}
			r.mu.Unlock()
		case <-r.shutdownCh:
			return
		case <-electionTimer:
			return
		default:
			<-time.After(5 * time.Millisecond)
			r.mu.Lock()
			if r.state != Candidate {
				r.mu.Unlock()
				return
			}
			r.mu.Unlock()
		}
	}
}

func (r *Raft) mainLoop() {
	for {
		r.mu.Lock()
		state := r.state
		r.mu.Unlock()

		switch state {
		case Candidate:
			r.candidateLoop()
		case Leader:
			r.leaderLoop()
		case Follower:
			r.followerLoop()
		case Shutdown:
			return
		}
	}
}

func (r *Raft) becomeCandidate() {
	// Expects mutex to be locked.
	r.state = Candidate
	r.options.logger.Infof("%s has entered the candidate state", r.id)
}

func (r *Raft) becomeLeader() {
	// Expects mutex to be locked.
	r.state = Leader
	for _, peer := range r.peers {
		peer.setNextIndex(r.log.LastIndex() + 1)
		peer.setMatchIndex(0)
	}
	r.options.logger.Infof("%s has entered the leader state", r.id)
}

func (r *Raft) becomeFollower(term uint64) {
	// Expects mutex to be locked.
	r.setCurrentTerm(term)
	r.votedFor = ""
	r.state = Follower
	r.options.logger.Infof("%s has entered the follower state", r.id)
}

func (r *Raft) setCurrentTerm(term uint64) {
	// Expects mutex to be locked.
	currentTermKey := []byte("currentTerm")
	if err := r.storage.SetUint64(currentTermKey, term); err != nil {
		r.options.logger.Fatalf("failed to persist current term to storage: %s", err.Error())
	}
	r.currentTerm = term
}

func (r *Raft) setVotedFor(votedFor string) {
	// Expects mutex to be locked.
	votedForKey := []byte("votedFor")
	if err := r.storage.Set(votedForKey, []byte(votedFor)); err != nil {
		r.options.logger.Fatalf("failed to persist vote to storage: %s", err.Error())
	}
	r.votedFor = votedFor
}

func (r *Raft) quorum() int {
	return len(r.peers)/2 + 1
}

func (r *Raft) status() (id string, term uint64, state State) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.id, r.currentTerm, r.state
}