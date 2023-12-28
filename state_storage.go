package raft

import (
	"io"
	"os"
	"path/filepath"

	"github.com/jmsadair/raft/internal/errors"
)

var errStateStorageNotOpen = errors.New("state storage is not open")

// StateStorage represents the component of Raft responsible for persistently storing
// term and vote.
type StateStorage interface {
	PersistentStorage

	// SetState persists the provided state. The storage must be open otherwise an
	// error is returned.
	SetState(term uint64, vote string) error

	// State returns the most recently persisted state in the storage. If there is
	// no pre-existing state or the storage is closed, zero and an empty string
	// will be returned. If the state storage is not open, an error will be returned.
	State() (uint64, string, error)
}

// persistentState is the state that must be persisted in Raft.
type persistentState struct {
	// The term of the associated Raft instance.
	term uint64

	// The vote of the associated Raft instance.
	votedFor string
}

// persistentStateStorage implements the StateStorage interface.
// This implementation is not concurrent safe.
type persistentStateStorage struct {
	// The directory where the state will be persisted.
	path string

	// The file associated with the storage, nil if storage is closed.
	file *os.File

	// The most recently persisted state.
	state persistentState
}

// NewStateStorage creates a new StateStorage at the provided path.
func NewStateStorage(path string) StateStorage {
	return &persistentStateStorage{path: path}
}

func (p *persistentStateStorage) Open() error {
	fileName := filepath.Join(p.path, "state.bin")
	file, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE, 0o666)
	if err != nil {
		return errors.WrapError(err, "failed to open state storage file")
	}
	p.file = file
	return nil
}

func (p *persistentStateStorage) Close() error {
	if p.file == nil {
		return nil
	}
	if err := p.file.Close(); err != nil {
		return errors.WrapError(err, "failed to close state storage file")
	}
	p.state = persistentState{}
	return nil
}

func (p *persistentStateStorage) Replay() error {
	if p.file == nil {
		return errStateStorageNotOpen
	}

	// Read the contents of the file associated with the storage.
	reader := io.Reader(p.file)
	state, err := decodePersistentState(reader)

	if err != nil && err != io.EOF {
		return errors.WrapError(err, "failed while replaying state storage")
	}

	p.state = state

	return nil
}

func (p *persistentStateStorage) SetState(term uint64, votedFor string) error {
	if p.file == nil {
		return errStateStorageNotOpen
	}

	// Create a temporary file that will replace the file currently associated with storage.
	// Note that it is NOT safe to truncate the file and then write the new state - it must
	// be atomic.
	tmpFile, err := os.CreateTemp(p.path, "tmp-")
	if err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}

	// Write the new state to the temporary file.
	p.state = persistentState{term: term, votedFor: votedFor}
	if err := encodePersistentState(tmpFile, &p.state); err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}
	if err := tmpFile.Sync(); err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}

	// Close the files to prepare for the rename.
	if err := tmpFile.Close(); err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}
	if err := p.file.Close(); err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}

	// Perform atomic rename to swap the newly persisted state with the old.
	if err := os.Rename(tmpFile.Name(), p.file.Name()); err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}

	// Open the state storage for future writes.
	fileName := filepath.Join(p.path, "state.bin")
	p.file, err = os.OpenFile(fileName, os.O_RDWR, 0o666)
	if err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}
	if _, err := p.file.Seek(0, io.SeekEnd); err != nil {
		return errors.WrapError(err, "failed while persisting state")
	}

	return nil
}

func (p *persistentStateStorage) State() (uint64, string, error) {
	if p.file == nil {
		return 0, "", errStateStorageNotOpen
	}
	return p.state.term, p.state.votedFor, nil
}
