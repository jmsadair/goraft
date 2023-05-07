package raft

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/jmsadair/raft/internal/errors"
)

const (
	errIndexDoesNotExist = "index %d does not exist"
)

// Log supports appending and retrieving log entries in
// in a durable manner.
type Log interface {
	Open() error

	Close() error

	// GetEntry returns the log entry located at index.
	GetEntry(index uint64) (*LogEntry, error)

	// AppendEntry appends a log entry to the log.
	AppendEntry(entry *LogEntry) error

	// AppendEntries appends multiple log entries to the log.
	AppendEntries(entries []*LogEntry) error

	// Truncate deletes all log entries with index greater
	// than or equal to the provided index.
	Truncate(index uint64) error

	// Compact deletes all log entries with index less than
	// or equal to the provided index.
	Compact(index uint64) error

	// Contains returns true if the index exists in the log and
	// false otherwise.
	Contains(index uint64) bool

	// LastIndex returns the largest index that exists in the log and zero
	// if the log is empty.
	LastIndex() uint64

	// LastTerm returns the largest term in the log and zero if the log
	// is empty.
	LastTerm() uint64

	// NextIndex returns the next index to append to the log.
	NextIndex() uint64
}

type LogEntry struct {
	index  uint64
	term   uint64
	offset int64
	data   []byte
}

func NewLogEntry(index uint64, term uint64, data []byte) *LogEntry {
	return &LogEntry{index: index, term: term, data: data}
}

func (e *LogEntry) IsConflict(other *LogEntry) bool {
	return e.index == other.index && e.term != other.term
}

type PersistentLog struct {
	entries    []*LogEntry
	file       *os.File
	path       string
	logEncoder LogEncoder
	logDecoder LogDecoder
}

func NewPersistentLog(path string, logEncoder LogEncoder, logDecoder LogDecoder) *PersistentLog {
	return &PersistentLog{path: path, logEncoder: logEncoder, logDecoder: logDecoder}
}

func (l *PersistentLog) Open() error {
	file, err := os.OpenFile(l.path, os.O_RDWR|os.O_CREATE, 0777)
	if err != nil {
		return err
	}
	l.file = file

	reader := bufio.NewReader(l.file)
	entries := make([]*LogEntry, 0)
	for {
		entry, err := l.logDecoder.Decode(reader)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		entries = append(entries, &entry)
	}

	if len(entries) == 0 {
		entry := NewLogEntry(0, 0, nil)
		writer := bufio.NewWriter(l.file)
		if err := l.logEncoder.Encode(writer, entry); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
		entries = append(entries, entry)
	}

	l.entries = entries
	return nil
}

func (l *PersistentLog) Close() error {
	if l.file == nil {
		return nil
	}
	return l.file.Close()
}

func (l *PersistentLog) GetEntry(index uint64) (*LogEntry, error) {
	if l.file == nil {
		return nil, fmt.Errorf("log is not open")
	}
	logIndex := index - l.entries[0].index
	lastIndex := l.entries[len(l.entries)-1].index
	if logIndex <= 0 || logIndex > lastIndex {
		return nil, errors.WrapError(nil, errIndexDoesNotExist, index)
	}
	entry := l.entries[logIndex]
	return entry, nil
}

func (l *PersistentLog) Contains(index uint64) bool {
	logIndex := index - l.entries[0].index
	lastIndex := l.entries[len(l.entries)-1].index
	return !(logIndex <= 0 || logIndex > lastIndex)
}

func (l *PersistentLog) AppendEntry(entry *LogEntry) error {
	return l.AppendEntries([]*LogEntry{entry})
}

func (l *PersistentLog) AppendEntries(entries []*LogEntry) error {
	if l.file == nil {
		return fmt.Errorf("log is not open")
	}

	writer := bufio.NewWriter(l.file)
	for _, entry := range entries {
		offset, err := l.file.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		entry.offset = offset
		if err := l.logEncoder.Encode(writer, entry); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
	}

	if err := l.file.Sync(); err != nil {
		return err
	}

	l.entries = append(l.entries, entries...)

	return nil
}

func (l *PersistentLog) Truncate(index uint64) error {
	logIndex := index - l.entries[0].index
	lastIndex := l.entries[len(l.entries)-1].index
	if logIndex <= 0 || logIndex > lastIndex {
		return errors.WrapError(nil, errIndexDoesNotExist, index)
	}
	size := l.entries[logIndex].offset
	if err := l.file.Truncate(size); err != nil {
		return err
	}
	l.entries = l.entries[:logIndex]
	l.file.Sync()
	return nil
}

func (l *PersistentLog) Compact(index uint64) error {
	logIndex := index - l.entries[0].index
	lastIndex := l.entries[len(l.entries)-1].index
	if logIndex <= 0 || logIndex > lastIndex {
		return errors.WrapError(nil, errIndexDoesNotExist, index)
	}

	newEntries := make([]*LogEntry, len(l.entries)-int(logIndex))
	copy(newEntries[:], l.entries[logIndex:])

	compactedFile, err := ioutil.TempFile(".", "raft-log")
	if err != nil {
		return err
	}

	writer := bufio.NewWriter(compactedFile)
	for _, entry := range newEntries {
		offset, err := l.file.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		entry.offset = offset
		if err := l.logEncoder.Encode(writer, entry); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
	}

	compactedFile.Sync()
	if err := os.Rename(compactedFile.Name(), l.path); err != nil {
		return err
	}

	l.file = compactedFile
	l.entries = newEntries

	return nil
}

func (l *PersistentLog) LastTerm() uint64 {
	return l.entries[len(l.entries)-1].term
}

func (l *PersistentLog) LastIndex() uint64 {
	return l.entries[len(l.entries)-1].index
}

func (l *PersistentLog) NextIndex() uint64 {
	return l.entries[len(l.entries)-1].index + 1
}
