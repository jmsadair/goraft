package raft

import (
	"bufio"
	"io"
	"os"

	"github.com/jmsadair/raft/internal/errors"
)

var (
	errSnapshotStoreNotOpen = errors.New("snapshot storage is not open")
)

// Snapshot represents a snapshot of the replicated state machine.
type Snapshot struct {
	// Last index included in the snapshot
	LastIncludedIndex uint64

	// Last term included in the snapshot.
	LastIncludedTerm uint64

	// State of the state machine.
	Data []byte
}

// NewSnapshot creates a new Snapshot instance with the provided state data
// from the replicated state machine.
func NewSnapshot(lastIncludedIndex uint64, lastIncludedTerm uint64, data []byte) *Snapshot {
	dataCopy := make([]byte, len(data))
	copy(dataCopy[:], data)
	return &Snapshot{
		LastIncludedIndex: lastIncludedIndex,
		LastIncludedTerm:  lastIncludedTerm,
		Data:              dataCopy,
	}
}

// SnapshotStorage represents the component of Raft this manages persistently
// storing snapshots of the state machine.
type SnapshotStorage interface {
	PersistentStorage

	// LastSnapshot gets the most recently saved snapshot if it exists.
	// If the snapshot does not exist, it will be nil. An error is returned
	// if the snapshot storage is not open.
	LastSnapshot() (*Snapshot, error)

	// SaveSnapshot persists the provided snapshot.
	SaveSnapshot(snapshot *Snapshot) error

	// ListSnapshots returns an array of the snapshots that have been persisted.
	// An error is returned if the snapshot storage is not open.
	ListSnapshots() ([]Snapshot, error)
}

// persistentSnapshotStorage is an implementation of the SnapshotStorage interface. This
// implementation is not concurrent safe.
type persistentSnapshotStorage struct {
	// All snapshots that have been persisted. This array is empty
	// if no snapshots have been persisted or the storage has not
	// been opened.
	snapshots []Snapshot

	// The path to where snapshots are persisted.
	path string

	// The file that the snapshots are persisted to. This value is nil if the storage
	// has not been opened.
	file *os.File
}

// NewSnapshotStorage creates a new instance of SnapshotStorage at the
// provided path.
func NewSnapshotStorage(path string) SnapshotStorage {
	return &persistentSnapshotStorage{path: path}
}

func (p *persistentSnapshotStorage) Open() error {
	if p.file != nil {
		return nil
	}

	file, err := os.OpenFile(p.path, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		return errors.WrapError(err, "failed to open snapshot storage")
	}

	p.file = file
	p.snapshots = make([]Snapshot, 0)

	return nil
}

func (p *persistentSnapshotStorage) Replay() error {
	if p.file == nil {
		return errSnapshotStoreNotOpen
	}

	reader := bufio.NewReader(p.file)

	for {
		snapshot, err := decodeSnapshot(reader)
		if err == io.EOF {
			break
		}
		if err != nil {
			return errors.WrapError(err, "failed while replaying snapshot storage")
		}
		p.snapshots = append(p.snapshots, snapshot)
	}

	return nil
}

func (p *persistentSnapshotStorage) Close() error {
	if p.file == nil {
		return nil
	}

	if err := p.file.Close(); err != nil {
		return errors.WrapError(err, "failed to close snapshot storage")
	}
	p.snapshots = nil
	p.file = nil

	return nil
}

func (p *persistentSnapshotStorage) LastSnapshot() (*Snapshot, error) {
	if p.file == nil {
		return nil, errSnapshotStoreNotOpen
	}
	if len(p.snapshots) == 0 {
		return nil, nil
	}
	return &p.snapshots[len(p.snapshots)-1], nil
}

func (p *persistentSnapshotStorage) ListSnapshots() ([]Snapshot, error) {
	if p.file == nil {
		return p.snapshots, errSnapshotStoreNotOpen
	}
	return p.snapshots, nil
}

func (p *persistentSnapshotStorage) SaveSnapshot(snapshot *Snapshot) error {
	if p.file == nil {
		return errSnapshotStoreNotOpen
	}

	writer := bufio.NewWriter(p.file)
	if err := encodeSnapshot(writer, snapshot); err != nil {
		return errors.WrapError(err, "failed to save snapshot")
	}
	if err := writer.Flush(); err != nil {
		return errors.WrapError(err, "failed to save snapshot")
	}
	if err := p.file.Sync(); err != nil {
		return errors.WrapError(err, "failed to save snapshot")
	}

	p.snapshots = append(p.snapshots, *snapshot)

	return nil
}
