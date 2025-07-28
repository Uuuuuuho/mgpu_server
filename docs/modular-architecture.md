# Multi-GPU Scheduler - New Modular Architecture

## Overview

The Multi-GPU Scheduler has been reorganized into a modular package structure for better maintainability, understanding, and extensibility. The new structure follows Python best practices and provides clear separation of concerns.

## New Package Structure

```
src/mgpu/
├── __init__.py                 # Main package with public API exports
├── cli.py                      # Command-line entry points
├── core/                       # Core data models and configuration
│   ├── __init__.py
│   ├── models.py              # SimpleJob, JobProcess, NodeInfo, MessageType
│   └── config.py              # Configuration management
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── logging.py             # Logging utilities
│   ├── network.py             # Network communication utilities
│   └── gpu.py                 # GPU detection and management
├── server/                     # Master server components
│   ├── __init__.py
│   └── master.py              # SimpleMaster class
├── client/                     # Client components
│   ├── __init__.py
│   └── client.py              # SimpleClient class
└── node/                       # Node agent components
    ├── __init__.py
    └── agent.py               # SimpleNode and NodeManager classes
```

## New Executable Scripts

The original monolithic files have been replaced with thin wrappers that use the new modular package:

- `mgpu_client.py` - Client interface
- `mgpu_server.py` - Master server  
- `mgpu_node.py` - Node agent
- `mgpu_queue.py` - Queue status (to be updated)
- `mgpu_cancel.py` - Job cancellation (to be updated)
- `mgpu_srun.py` - SLURM-like interface (to be updated)

## Key Benefits

### 1. Modular Design
- **Core Models**: Data structures separated into `core/models.py`
- **Configuration**: Centralized in `core/config.py`
- **Utilities**: Reusable components in `utils/`
- **Components**: Server, client, and node logic separated

### 2. Better Maintainability
- Smaller, focused files instead of large monolithic scripts
- Clear dependencies and imports
- Separation of concerns

### 3. Improved Testing
- Individual components can be tested in isolation
- Mock objects easier to create for testing
- Better unit test coverage possible

### 4. Enhanced Extensibility
- New features can be added to specific modules
- Plugin architecture possible
- API clearly defined in `__init__.py`

## Usage Examples

### Using the Package API

```python
from mgpu import SimpleClient, SimpleMaster, SimpleNode

# Create and use client
client = SimpleClient('localhost', 8080)
client.submit_job(gpus=2, cmd='python train.py')

# Create and start server
server = SimpleMaster('0.0.0.0', 8080)
server.start_server()

# Create and start node
node = SimpleNode('node1', 'localhost', 8080, 8081)
node.start_agent()
```

### Using the Command Line

```bash
# Start server
python src/mgpu_server.py --host 0.0.0.0 --port 8080

# Start node
python src/mgpu_node.py --node-id node1 --master-host localhost

# Submit job
python src/mgpu_client.py --gpus 2 'python train.py'
```

## Migration from Original Structure

### Original Files → New Modules

| Original File | New Module | Main Class |
|---------------|------------|------------|
| `mgpu_simple_master.py` | `mgpu.server.master` | `SimpleMaster` |
| `mgpu_simple_client.py` | `mgpu.client.client` | `SimpleClient` |
| `mgpu_simple_node.py` | `mgpu.node.agent` | `SimpleNode` |

### Preserved Functionality

All original functionality has been preserved:
- Job submission and execution
- Interactive sessions with real-time output
- GPU allocation and management
- Node registration and heartbeats
- Queue management and status
- Job cancellation
- Timeout configuration
- Error handling

### New Features

1. **Command-line Interface**: Unified CLI with multiple entry points
2. **Configuration Management**: Environment variable support
3. **Improved Logging**: Centralized logging setup
4. **Node Manager**: Support for managing multiple node agents
5. **Public API**: Clean imports from main package

## Development Guidelines

### Adding New Features

1. **Core Models**: Add new data structures to `core/models.py`
2. **Configuration**: Add new settings to `core/config.py`
3. **Utilities**: Add reusable functions to appropriate `utils/` modules
4. **Components**: Add new functionality to `server/`, `client/`, or `node/`

### Testing

```bash
# Test individual components
python -m pytest test/test_core_models.py
python -m pytest test/test_client.py
python -m pytest test/test_server.py

# Run all tests
python -m pytest test/
```

### Import Structure

- Use relative imports within the package: `from ..core.models import SimpleJob`
- Import from main package in applications: `from mgpu import SimpleClient`
- Avoid circular imports by keeping utilities separate

## Compatibility

The new modular structure maintains full compatibility with existing:
- PyTorch test scripts
- Timeout configurations
- Job submission workflows
- Interactive sessions
- Multi-node setups

## Future Enhancements

The modular structure enables:
- Plugin system for custom schedulers
- Multiple backend support (SLURM, Kubernetes)
- Advanced scheduling algorithms
- Web-based management interface
- Metrics collection and monitoring
- Auto-scaling capabilities
