# audio-capture

CLI en Swift que captura audio del sistema usando Core Audio Taps (macOS 14.2+) y escribe a stdout PCM raw (int16, little-endian, mono, 16kHz configurable). Usa solo el permiso de **Audio Capture**, no Screen Recording.

## Requisitos

- macOS 14.2+ (Core Audio Taps)
- Swift 5.7+
- Permisos de Audio Capture (Screen & System Audio Recording)

## Instalación

### Compilar desde código fuente

```bash
cd swift-audio-capture
swift build -c release
```

El ejecutable se encontrará en `.build/release/audio-capture`.

**Nota**: Si la compilación falla por permisos de caché o por desajuste de versión del SDK (macOS 26 / Tahoe), prueba desde Xcode o asegúrate de que el toolchain de Swift coincida con el SDK. También puedes compilar desde una terminal fuera del IDE.

### Instalar en PATH (opcional)

```bash
cp .build/release/audio-capture /usr/local/bin/
```

## Uso

### Captura continua

Captura audio del sistema y escribe PCM raw a stdout:

```bash
audio-capture --sample-rate 16000 --channels 1
```

### Comprobar permiso de Audio Capture

```bash
audio-capture --list
```

### Opciones

- `--sample-rate <rate>`: Tasa de muestreo en Hz (por defecto: 16000)
- `--channels <count>`: Número de canales (por defecto: 1, mono)
- `--list`: Comprueba si el permiso de Audio Capture está concedido
- `--help, -h`: Muestra ayuda

## Output

El programa escribe datos PCM raw (int16, little-endian) directamente a stdout. No incluye headers WAV ni metadatos.

### Formato de salida

- **Formato**: PCM raw
- **Tipo de datos**: int16 (16-bit signed integer)
- **Endianness**: Little-endian
- **Canales**: Mono (por defecto) o estéreo si se especifica `--channels 2`
- **Sample rate**: Configurable (por defecto: 16000 Hz)

### Ejemplo de uso desde Python

```python
import subprocess

process = subprocess.Popen(
    ['audio-capture', '--sample-rate', '16000'],
    stdout=subprocess.PIPE,
    bufsize=0
)

# Leer chunks de PCM
while True:
    chunk = process.stdout.read(32000)  # 1 segundo de audio a 16kHz mono
    if not chunk:
        break
    # Procesar con whisper u otro procesador de audio...
```

## Permisos

El programa requiere permisos de **Audio Capture** (no Screen Recording):

1. La primera vez que ejecutes el programa, macOS puede mostrar un diálogo
2. O ve manualmente a: **System Settings > Privacy & Security > Screen & System Audio Recording**
3. Agrega tu terminal (Terminal, iTerm, etc.)

**Nota**: Solo se pide acceso al audio del sistema, no a la pantalla.

## Detalles Técnicos

### Core Audio Taps

Este programa usa la API **Core Audio Taps** (`AudioHardwareCreateProcessTap`, macOS 14.2+) para capturar audio del sistema sin necesitar el permiso de Screen Recording. No requiere dispositivos virtuales como BlackHole.

### Conversión de Audio

- Core Audio Taps entrega audio en formato float32 (-1.0 a 1.0)
- El programa convierte automáticamente a int16 PCM
- Si el audio es estéreo, se mezcla automáticamente a mono (promedio de canales)
- El resampling se realiza si el sample rate del sistema difiere del configurado

### Manejo de Señales

El programa maneja correctamente:
- `SIGINT` (Ctrl+C): Detiene la captura limpiamente
- `SIGTERM`: Detiene la captura limpiamente

## Troubleshooting

### Error: "Audio Capture permission not granted"

1. Ve a **System Settings > Privacy & Security > Screen & System Audio Recording**
2. Asegúrate de que tu terminal esté en la lista y habilitada
3. Reinicia el programa

### No se captura audio

- Verifica que haya audio reproduciéndose en el sistema
- Algunas aplicaciones pueden bloquear la captura de audio
- Prueba con `audio-capture --list` para verificar el permiso

### El programa se cierra inmediatamente

- Verifica los logs en stderr para ver el error específico
- Asegúrate de tener macOS 14.2+ (verifica con `sw_vers`)

## Desarrollo

### Estructura del Proyecto

```
swift-audio-capture/
├── Package.swift                    # Configuración del paquete Swift
├── Info.plist                       # NSAudioCaptureUsageDescription
├── Sources/
│   └── audio-capture/
│       ├── main.swift               # Punto de entrada del CLI
│       ├── CoreAudioTap.swift       # Wrapper Swift para Core Audio Taps
│       ├── CoreAudioTapBridge.h/m   # Bridge Obj-C (CATapDescription, etc.)
│       └── PCMConverter.swift       # Conversión a int16
└── README.md                        # Este archivo
```

### Compilar y ejecutar

```bash
# Desarrollo
swift build
swift run audio-capture --sample-rate 16000

# Release
swift build -c release
.build/release/audio-capture --sample-rate 16000
```

## Licencia

Ver LICENSE en el directorio raíz del proyecto.
