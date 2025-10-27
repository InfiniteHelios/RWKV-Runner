# RWKV-Runner Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Backend Systems](#backend-systems)
5. [Frontend Architecture](#frontend-architecture)
6. [Deployment & Distribution](#deployment--distribution)
7. [API Documentation](#api-documentation)
8. [Model Management](#model-management)
9. [MIDI Support](#midi-support)
10. [Development Workflow](#development-workflow)

---

## Project Overview

RWKV-Runner is a comprehensive desktop application framework that provides an all-in-one solution for running RWKV (Recurrent Weighted Key-Value Transformer) language models. The project automates the entire process of model setup, inference, and management while providing an OpenAI-compatible API interface.

### Key Features

- **Model Management**: Automatic download, conversion, and deployment of RWKV models
- **Multi-Backend Support**: Python-based inference with WebGPU/Rust backend options
- **Desktop GUI**: Native application built with Wails (Go + React)
- **OpenAI Compatibility**: Full compatibility with OpenAI API for ChatGPT integration
- **MIDI Support**: Real-time MIDI hardware input for music generation
- **LoRA Fine-tuning**: Built-in LoRA fine-tuning capabilities (Windows)
- **Cross-Platform**: Supports Windows, macOS, and Linux
- **WebUI Option**: Optional web-based interface

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Desktop Application (Wails)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Frontend    │  │  Go Backend  │  │  File System │     │
│  │  (React)     │  │  Controller  │  │  Watcher     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Backend (FastAPI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  RWKV    │  │  MIDI    │  │  Config  │  │  Misc    │   │
│  │  Routes  │  │  Routes  │  │  Routes  │  │  Routes  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┴──────┐  │
│                                                          │  │
│  ┌───────────────────────────────────────────────────┐ │  │
│  │         RWKV Model Inference Engine                │ │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │ │  │
│  │  │  RWKV-4  │  │  RWKV-5  │  │  RWKV-6  │         │ │  │
│  │  │  (Raven) │  │  (World) │  │  (Music) │         │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘         │ │  │
│  └───────────────────────────────────────────────────┘ │  │
└────────────────────────────────────────────────────────┴──┘
```

### Component Interaction Flow

```
User Action → Frontend → Go Bridge → Python Backend → RWKV Model
                      ↓                                        ↓
                 File System                              Generate
                   Watcher                                  Response
                      ↓                                        ↓
                 Event Emit                              Backend Response
                      ↓                                        ↓
                   Frontend ←─────────────────────────── JSON Response
```

---

## Core Components

### 1. Go Backend (`backend-golang/`)

The Go backend serves as the orchestrator and middleware between the frontend and Python inference backend.

#### Key Files

- **`app.go`**: Main application struct with file watching, hardware monitoring, and update management
- **`rwkv.go`**: RWKV-specific operations (model conversion, server management, LoRA merging)
- **`midi.go`**: MIDI hardware input handling
- **`download.go`**: Model download management
- **`file.go`**: File operations and utilities
- **`utils.go`**: Platform-specific utilities (WSL support, etc.)

#### Responsibilities

- **Application Lifecycle**: Startup, shutdown, and restart management
- **File System Monitoring**: Watch model directories for changes using `fsnotify`
- **Hardware Monitoring**: Real-time system resource monitoring (Windows only)
- **Process Management**: Launch and manage Python backend processes
- **Proxy Management**: HTTP proxy for resource fetching
- **Auto-Update**: Self-update mechanism with rollback support

#### Key Structures

```go
type App struct {
    ctx           context.Context
    HasConfigData bool
    ConfigData    map[string]any
    Dev           bool
    proxyPort     int
    exDir         string
    cmdPrefix     string
}
```

### 2. Python Backend (`backend-python/`)

The Python backend handles all model inference operations using PyTorch and RWKV libraries.

#### Key Files

- **`main.py`**: FastAPI application entry point
- **`utils/rwkv.py`**: Core RWKV model implementation with multi-version support
- **`utils/torch.py`**: PyTorch device management and CUDA configuration
- **`utils/midi.py`**: MIDI tokenization and processing
- **`routes/completion.py`**: OpenAI-compatible completion endpoints
- **`routes/config.py`**: Model configuration management
- **`routes/midi.py`**: MIDI-specific API endpoints
- **`routes/state_cache.py`**: Context state caching
- **`global_var.py`**: Global state management

#### Model Types Supported

- **RWKV-4 (Raven)**: Original RWKV model with RNN-like behavior
- **RWKV-5 (World)**: Enhanced version with improved multilingual support
- **RWKV-6 (Music)**: Specialized for music generation with MIDI tokenization

#### Key Classes

```python
class AbstractRWKV(ABC):
    """Base class for RWKV model implementations"""
    
    def get_embedding(self, input: str, fast_mode: bool) -> Tuple[List[float], int]
    def run_rnn(self, tokens: List[str]) -> Tuple[List[float], int]
    def fix_tokens(self, tokens) -> List[int]
```

### 3. Frontend (`frontend/`)

React-based desktop application with Fluent UI components.

#### Technology Stack

- **React 18**: UI framework
- **TypeScript**: Type safety
- **MobX**: State management
- **React Router**: Navigation
- **Fluent UI**: Microsoft's design system
- **Tailwind CSS**: Utility-first styling
- **Vite**: Build tool

#### Key Directories

- **`src/pages/`**: Main application pages
  - Home, Chat, Completion, Composition, Configs, Models, Downloads, Train, Settings, About
- **`src/components/`**: Reusable UI components
- **`src/stores/`**: MobX state stores
- **`src/utils/`**: Utility functions
- **`src/types/`**: TypeScript type definitions
- **`src/apis/`**: API client functions

#### State Management

```typescript
// MobX store pattern
class CommonStore {
  @observable settings: Settings
  @observable currentModel: string
  @observable hardwareInfo: HardwareInfo
}
```

### 4. Model Conversion Tools

#### Supported Formats

- **Safetensors → RWKV**: Safe TensorFlow format conversion
- **PyTorch → GGML**: Quantized GGML format for CPU inference
- **Custom Quantization**: Q5_1 and FP16 quantization support

#### Conversion Scripts

- `backend-python/convert_model.py`: Main conversion orchestrator
- `backend-python/convert_safetensors.py`: Safetensors handler
- `backend-python/convert_pytorch_to_ggml.py`: GGML converter
- `backend-rust/web-rwkv-converter`: Rust-based converter for WebGPU

### 5. Fine-tuning System (`finetune/`)

#### LoRA (Low-Rank Adaptation) Support

- **V4 LoRA**: RWKV v4 fine-tuning
- **V5 LoRA**: RWKV v5 fine-tuning
- **V6 LoRA**: RWKV v6 fine-tuning with advanced features

#### Components

- **Data Preprocessing**: `json2binidx_tool/` for data indexing
- **Training Scripts**: `lora/v6/train.py` and variants
- **Model Merging**: `lora/merge_lora.py` for merging LoRA adapters
- **CUDA Kernels**: Custom CUDA operations for efficient training

#### Training Data Format

```json
{"text": "Training sample text content"}
```

---

## Backend Systems

### 1. FastAPI Server (`main.py`)

#### Middleware

- **CORS**: Cross-origin resource sharing enabled for all origins
- **Logging**: Request logging middleware
- **Global State**: Per-request state management

#### API Routes

```python
app.include_router(completion.router)    # /chat/completions, /completions, etc.
app.include_router(config.router)        # /switch-model, /load-model-config
app.include_router(midi.router)          # /midi/tokenize, /midi/detokenize
app.include_router(file_process.router)  # /upload, /read-state
app.include_router(misc.router)         # /health, /hardware-info
app.include_router(state_cache.router)   # /save-state, /load-state
```

#### Lifespan Management

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init()  # Initialize global state
    yield
```

### 2. RWKV Engine (`utils/rwkv.py`)

#### Abstract Base Implementation

```python
class AbstractRWKV(ABC):
    """Core RWKV interface"""
    
    # Token adjustment methods
    def adjust_occurrence(self, occurrence: Dict, token: int)
    def adjust_forward_logits(self, logits: List[float], occurrence: Dict, i: int)
    def fix_tokens(self, tokens) -> List[int]
    
    # Inference methods
    def run_rnn(self, tokens: List[str]) -> Tuple[List[float], int]
    def delta_postprocess(self, delta: str) -> str
    
    # Embedding generation
    def get_embedding(self, input: str, fast_mode: bool) -> Tuple[List[float], int]
```

#### Model Version Handlers

- `RWKV-4`: Standard RNN-like processing
- `RWKV-5`: Enhanced with frequency penalties and occurrence tracking
- `RWKV-6`: Music-specific token processing

#### Custom CUDA Operations

- **WKV Operation**: Weighted key-value computation for RWKV-5/6
- **Element-wise Operations**: Optimized CUDA kernels for model inference
- **Location**: `backend-python/rwkv_pip/cuda/`

### 3. Device Management (`utils/torch.py`)

#### GPU Strategy Configuration

```python
STRATEGY_PRESETS = {
    "cpu fp32": CPU_FP32,
    "cuda fp16": CUDA_FP16,
    "cuda fp16*i8": CUDA_FP16_I8,
    "cuda fp16*i4": CUDA_FP16_I4,
    "webgpu fp16": WebGPU_FP16,
}
```

#### Features

- **Auto-detection**: Automatically detect CUDA/CPU availability
- **Memory Management**: Efficient VRAM usage with quantization
- **Custom Kernels**: Optional custom CUDA kernels for acceleration
- **WebGPU Support**: Browser-based GPU acceleration via Rust backend

---

## Frontend Architecture

### Page Structure

```
src/
├── pages/
│   ├── Home.tsx           # Model selection and status
│   ├── Chat.tsx            # Conversational interface
│   ├── Completion.tsx      # Text completion
│   ├── Composition.tsx    # MIDI-based music generation
│   ├── Configs.tsx        # Model configuration
│   ├── Models.tsx         # Model management
│   ├── Downloads.tsx      # Download management
│   ├── Train.tsx          # LoRA fine-tuning (Windows)
│   ├── Settings.tsx       # Application settings
│   └── About.tsx          # About page
```

### Component Hierarchy

```
App.tsx
├── FluentProvider (theme)
├── Navigation (vertical tabs)
└── Page Routes
    ├── Page Component
    │   ├── Section
    │   │   ├── Labeled (form fields)
    │   │   └── ValuedSlider (numeric input)
    │   └── Action Buttons
    └── CustomToastContainer
```

### State Flow

```
User Input → Component → MobX Store → Go Bridge → Python Backend
                                        ↓
                                   File Watcher
                                        ↓
                                   Event Emit
                                        ↓
                                   Store Update
                                        ↓
                                   Component Re-render
```

### Key Frontend Features

#### 1. Chat Interface (`Chat.tsx`)

- **Streaming Responses**: Server-sent events for real-time streaming
- **Message History**: Context-aware conversation management
- **Presets**: Conversation templates
- **Attachment Support**: File upload and processing

#### 2. Composition Interface (`Composition.tsx`)

- **MIDI Input**: Real-time MIDI keyboard support
- **Track Editing**: Visual MIDI track editor
- **Playback**: Built-in MIDI player with soundfont support
- **Export**: MIDI file export functionality

#### 3. Model Management (`Models.tsx`)

- **Model Discovery**: Automatic model detection
- **Conversion**: Model format conversion interface
- **Remote Inspection**: HuggingFace model browsing
- **State Management**: Model state persistence

---

## Deployment & Distribution

### Build System

#### Wails Configuration (`wails.json`)

```json
{
  "$schema": "https://wails.io/schemas/config.json",
  "name": "RWKV-Runner",
  "outputfilename": "RWKV-Runner",
  "frontend": {"dir": "./frontend", "install": "npm install"}
}
```

#### Embedded Resources

- Frontend build artifacts
- Python runtime (Python 3.10 embedded)
- PyTorch and dependencies
- Custom CUDA libraries
- WebGPU backend (Rust)
- Fine-tuning tools
- MIDI assets and soundfonts

### Distribution Targets

#### Windows (`build/windows/`)

- **Executable**: `RWKV-Runner.exe`
- **Installer**: NSIS-based installer (`installer/project.nsi`)
- **Embedded Python**: `py310/` directory
- **Components**: Hardware monitoring tools

#### macOS (`build/darwin/`)

- **App Bundle**: `RWKV-Runner.app`
- **Signing**: Code signing with entitlements
- **Info.plist**: Application metadata

#### Linux (`build/linux/`)

- **Binary**: Single executable
- **Dependencies**: Runtime dependencies documentation

### Auto-Update Mechanism

#### Self-Update Process

1. Check for updates via GitHub releases
2. Download update package (zip/tar.gz)
3. Extract and apply using `selfupdate` library
4. Restart application
5. Rollback on failure

---

## API Documentation

### OpenAI-Compatible Endpoints

#### Chat Completions

```http
POST /chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "model": "default",
  "temperature": 1.0,
  "max_tokens": 500
}
```

#### Embeddings

```http
POST /embeddings
Content-Type: application/json

{
  "input": "text to embed",
  "model": "default"
}
```

#### Completions

```http
POST /completions
Content-Type: application/json

{
  "prompt": "Complete this sentence:",
  "temperature": 1.0,
  "max_tokens": 100
}
```

### Model Management Endpoints

#### Switch Model

```http
POST /switch-model
Content-Type: application/json

{
  "model": "models/my-model.rwkv",
  "strategy": "cuda fp16"
}
```

#### Model Configuration

```http
POST /load-model-config
Content-Type: application/json

{
  "config": {...}
}
```

### MIDI Endpoints

#### Tokenize MIDI

```http
POST /midi/tokenize
Content-Type: application/json

{
  "midi_path": "path/to/file.mid",
  "tokenizer": "midi"
}
```

#### Detokenize to MIDI

```http
POST /midi/detokenize
Content-Type: application/json

{
  "tokens": [123, 456, ...],
  "tokenizer": "midi"
}
```

### State Management

#### Save State

```http
POST /save-state
Content-Type: application/json

{
  "path": "state-models/my-state.pth"
}
```

#### Load State

```http
POST /load-state
Content-Type: application/json

{
  "path": "state-models/my-state.pth"
}
```

---

## Model Management

### Model Types

#### 1. RWKV-4 (Raven)

- **Architecture**: Recurrent architecture with attention mechanism
- **Use Cases**: General purpose text generation
- **Advantages**: Fast inference, low memory usage

#### 2. RWKV-5 (World)

- **Architecture**: Enhanced RWKV with improved multilingual capabilities
- **Use Cases**: Multilingual text generation
- **Advantages**: Better multilingual understanding

#### 3. RWKV-6 (Music)

- **Architecture**: Music-specific tokenization and processing
- **Use Cases**: Music generation, MIDI composition
- **Advantages**: Optimized for musical sequences

### Model Conversion Pipeline

```
Raw Model → Format Detection → Conversion → Deployment
    ↓              ↓              ↓            ↓
 Safetensors    Strategy      Quantization   Load
 PyTorch        Selection     Optimization   Runtime
```

### Download System

#### Features

- **Automatic Discovery**: Browse HuggingFace models
- **Resume Support**: Resume interrupted downloads
- **Progress Tracking**: Real-time download progress
- **Format Detection**: Automatic model format detection

#### Implementation

```go
func (a *App) DownloadFile(url string, destination string) (string, error)
func (a *App) GetDownloadStatus() []DownloadStatus
```

---

## MIDI Support

### Hardware Input

#### Supported Platforms

- **Windows**: USB MIDI via WinMM
- **macOS**: USB + Bluetooth MIDI (via Bluetooth MIDI Connect)
- **Linux**: ALSA MIDI support

#### Connection Methods

1. **USB MIDI**: Direct plug-and-play
2. **Bluetooth MIDI** (macOS): Native support
3. **Bluetooth MIDI** (Windows): Via loopMIDI + MIDIberry

### MIDI Processing

#### Tokenization

- **MIDI-LLM Tokenizer**: Converts MIDI to sequence tokens
- **Piano Roll Format**: Note representation
- **Real-time Processing**: Streaming MIDI input

#### Token Types

```python
MIDI_EVENT_TYPES = {
    'note_on': ...,
    'note_off': ...,
    'control_change': ...,
    'tempo_change': ...,
}
```

### MIDI Generation Flow

```
MIDI Input → Tokenization → Model Inference → Detokenization → MIDI Output
                                       ↓
                                 Music Tokens
                                       ↓
                                  Control Flow
```

---

## Development Workflow

### Prerequisites

```bash
# Required
- Go 1.20+
- Node.js 18+
- Python 3.10
- PyTorch (CUDA optional)
- Wails CLI

# Optional
- CUDA Toolkit (for GPU acceleration)
- Rust (for WebGPU backend)
```

### Development Setup

#### 1. Clone and Install

```bash
git clone https://github.com/josStorer/RWKV-Runner
cd RWKV-Runner

# Install Wails CLI
go install github.com/wailsapp/wails/v2/cmd/wails@latest

# Install frontend dependencies
cd frontend
npm install
cd ..

# Install Python dependencies
cd backend-python
pip install -r requirements.txt
```

#### 2. Development Mode

```bash
# Terminal 1: Go backend
wails dev

# Terminal 2: Python backend (if needed)
cd backend-python
python main.py --dev

# Terminal 3: Frontend (optional, for hot-reload)
cd frontend
npm run dev
```

### Build Process

#### Development Build

```bash
wails build -dev
```

#### Production Build

```bash
# Windows
wails build -platform windows/amd64

# macOS
wails build -platform darwin/universal

# Linux
wails build -platform linux/amd64
```

### File Structure

```
RWKV-Runner/
├── main.go                 # Wails entry point
├── backend-golang/         # Go backend
├── backend-python/         # Python inference backend
├── backend-rust/           # WebGPU backend (optional)
├── frontend/               # React UI
├── finetune/               # LoRA fine-tuning
├── assets/                 # Static assets
├── build/                  # Build configurations
└── deploy-examples/        # Deployment examples
```

---

## Technical Stack Summary

### Backend

- **Go**: Application orchestrator and GUI framework
- **Python**: Model inference and API server
- **Rust**: WebGPU backend (optional)
- **C++/CUDA**: Custom model kernels

### Frontend

- **React 18**: UI framework
- **TypeScript**: Type safety
- **MobX**: State management
- **Fluent UI**: Design system
- **Tailwind CSS**: Styling

### Libraries & Frameworks

- **Wails v2**: Desktop application framework
- **FastAPI**: Python API server
- **PyTorch**: Model inference
- **uvicorn**: ASGI server
- **fsnotify**: File system watching

### Build Tools

- **Vite**: Frontend build tool
- **Wails Build**: Desktop build system
- **NSIS**: Windows installer
- **Embed**: Go embed for resources

---

## Security Considerations

### API Security

- CORS enabled for local development
- Production: Implement authentication
- Rate limiting recommended for public deployments
- Request size limits to prevent resource exhaustion

### File System

- Sandboxed model directories
- State file validation
- Safe file path handling

### Updates

- Code signature verification
- Rollback mechanism
- Secure update distribution

---

## Performance Optimization

### Model Inference

- **Custom CUDA Kernels**: Up to 50% speedup
- **Quantization**: Q5_1 quantization for CPU
- **State Caching**: Context state persistence
- **Batch Processing**: Efficient token batching

### Memory Management

- **Strategy Presets**: Multi-level VRAM configs
- **Dynamic Loading**: On-demand model loading
- **State Offloading**: State file management

### Frontend

- **Code Splitting**: Lazy loading of pages
- **Virtualization**: Large list rendering
- **Memoization**: Expensive computation caching

---

## Future Development

### Planned Features

- Enhanced fine-tuning UI
- Model marketplace integration
- Advanced MIDI features
- Distributed inference
- Multi-model ensemble

### Extensibility

- Plugin system for custom backends
- Custom tokenizer support
- Export formats
- API extensions

---

## Conclusion

RWKV-Runner provides a comprehensive, production-ready solution for running RWKV language models with an intuitive desktop interface and full OpenAI API compatibility. The architecture supports multiple model versions, hardware acceleration options, and specialized features like MIDI generation and LoRA fine-tuning.

### Key Strengths

1. **User-Friendly**: One-click model setup and deployment
2. **Flexible**: Multiple backend options and strategies
3. **Compatible**: Full OpenAI API compatibility
4. **Feature-Rich**: Chat, completion, composition, fine-tuning
5. **Cross-Platform**: Windows, macOS, Linux support

This technical documentation should serve as a comprehensive reference for understanding, developing, and extending the RWKV-Runner platform.
