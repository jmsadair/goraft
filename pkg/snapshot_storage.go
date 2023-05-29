package raft

import (
	"bufio"
	"io"
	"os"

	"github.com/jmsadair/raft/internal/errors"
)

// Snapshot represents a snapshot of the replicated state machine.
type Snapshot struct {
	// Last index included in snapshot
	LastIncludedIndex uint64
	// Last term included in snapshot.
	LastIncludedTerm uint64
	// State of replicated state machine.
	Data []byte
}

// NewSnapshot creates a new Snapshot instance with the provided values.
//
// Parameters:
//   - lastIncludedIndex: The last index included in the snapshot.
//   - lastIncludedTerm: The last term included in the snapshot.
//   - data: The state data of the replicated state machine.
//
// Returns:
//   - *Snapshot: A pointer to the created Snapshot instance.
func NewSnapshot(lastIncludedIndex uint64, lastIncludedTerm uint64, data []byte) *Snapshot {
	return &Snapshot{LastIncludedIndex: lastIncludedIndex, LastIncludedTerm: lastIncludedTerm, Data: data}
}

// SnapshotStorage is an interface representing a component responsible for managing snapshots.
type SnapshotStorage interface {
	Open() error

	Close() error

	// LastSnapshot gets the most recently saved snapshot, if it exists.
	//
	// Returns:
	//     - Snapshot: The most recently saved snapshot.
	//     - bool: True if the snapshot is valid and false otherwise.
	LastSnapshot() (Snapshot, bool)

	// SaveSnapshot saves the provided snapshot to durable storage.
	//
	// Parameters:
	//     - snapshot: The snapshot to be saved.
	//
	// Returns:
	//     - error: An error if saving the snapshot fails, or nil otherwise.
	SaveSnapshot(snapshot *Snapshot) error

	// ListSnapshots returns an array of the snapshots that have been saved.
	//
	// Returns:
	//     - []Snapshot: An array of the saved snapshots.
	ListSnapshots() []Snapshot
}

type PersistentSnapshotStorage struct {
	snapshots []Snapshot
	path      string
	file      *os.File
	encoder   SnapshotEncoder
	decoder   SnapshotDecoder
}

func NewPersistentSnapshotStorage(path string, encoder SnapshotEncoder, decoder SnapshotDecoder) *PersistentSnapshotStorage {
	return &PersistentSnapshotStorage{path: path, encoder: encoder, decoder: decoder}
}

func (p *PersistentSnapshotStorage) Open() error {
	if p.file != nil {
		return nil
	}

	file, err := os.OpenFile(p.path, os.O_RDWR|os.O_CREATE, 0777)
	if err != nil {
		return errors.WrapError(err, "failed to open snapshot storage file: %s", err.Error())
	}

	p.file = file

	reader := bufio.NewReader(p.file)
	p.snapshots = make([]Snapshot, 0)
	for {
		snapshot, err := p.decoder.Decode(reader)
		if err == io.EOF {
			break
		}
		if err != nil {
			return errors.WrapError(err, "failed to recover snapshot storage: %s", err.Error())
		}
		p.snapshots = append(p.snapshots, snapshot)
	}

	return nil
}

func (p *PersistentSnapshotStorage) Close() error {
	if p.file == nil {
		return nil
	}

	if err := p.file.Close(); err != nil {
		return errors.WrapError(err, "failed to close snapshot store")
	}

	p.snapshots = nil
	p.file = nil

	return nil
}

func (p *PersistentSnapshotStorage) LastSnapshot() (Snapshot, bool) {
	if len(p.snapshots) == 0 {
		return Snapshot{}, false
	}
	return p.snapshots[len(p.snapshots)-1], true
}

func (p *PersistentSnapshotStorage) ListSnapshots() []Snapshot {
	if len(p.snapshots) == 0 {
		return []Snapshot{}
	}
	return p.snapshots
}

func (p *PersistentSnapshotStorage) SaveSnapshot(snapshot *Snapshot) error {
	if p.file == nil {
		return errors.WrapError(nil, "failed to save snapshot: snapshot storage not open: path = %s", p.path)
	}

	writer := bufio.NewWriter(p.file)
	if err := p.encoder.Encode(writer, snapshot); err != nil {
		return errors.WrapError(err, "failed to save snapshot: %s", err.Error())
	}
	if err := writer.Flush(); err != nil {
		return errors.WrapError(err, "failed to save snapshot: %s", err.Error())
	}

	p.file.Sync()

	p.snapshots = append(p.snapshots, *snapshot)

	return nil
}
