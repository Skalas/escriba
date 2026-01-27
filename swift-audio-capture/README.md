# audio-capture

CLI en Swift que captura audio del sistema usando ScreenCaptureKit y lo escribe a stdout como PCM raw (int16, little-endian, mono, 16kHz configurable).

## Requisitos

- macOS 12.3+ (ScreenCaptureKit requiere macOS Monterey o superior)
- Xcode 13+ y Swift 5.5+
- Permisos de Screen Recording

## Instalación

### Compilar desde código fuente

```bash
cd swift-audio-capture
swift build -c release
```

El ejecutable se encontrará en `.build/release/audio-capture`.

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

### Listar displays disponibles

```bash
audio-capture --list
```

### Opciones

- `--sample-rate <rate>`: Tasa de muestreo en Hz (por defecto: 16000)
- `--channels <count>`: Número de canales (por defecto: 1, mono)
- `--list`: Lista displays/dispositivos disponibles
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

El programa requiere permisos de **Screen Recording** en macOS:

1. La primera vez que ejecutes el programa, macOS mostrará un diálogo
2. O ve manualmente a: **System Settings > Privacy & Security > Screen Recording**
3. Agrega tu terminal (Terminal, iTerm, etc.)

**Nota**: No se requieren permisos de accesibilidad (a diferencia de soluciones como BlackHole).

## Detalles Técnicos

### ScreenCaptureKit

Este programa usa la API nativa de macOS `ScreenCaptureKit` para capturar audio del sistema, similar a como lo hace Notion AI. No requiere dispositivos virtuales como BlackHole.

### Conversión de Audio

- ScreenCaptureKit típicamente entrega audio en formato float32 (-1.0 a 1.0)
- El programa convierte automáticamente a int16 PCM
- Si el audio es estéreo, se mezcla automáticamente a mono (promedio de canales)
- El resampling se realiza si el sample rate del sistema difiere del configurado

### Manejo de Señales

El programa maneja correctamente:
- `SIGINT` (Ctrl+C): Detiene la captura limpiamente
- `SIGTERM`: Detiene la captura limpiamente

## Troubleshooting

### Error: "Screen Recording permission denied"

1. Ve a **System Settings > Privacy & Security > Screen Recording**
2. Asegúrate de que tu terminal esté en la lista y habilitada
3. Reinicia el programa

### No se captura audio

- Verifica que haya audio reproduciéndose en el sistema
- Algunas aplicaciones pueden bloquear la captura de audio
- Prueba con `--list` para verificar que los displays estén disponibles

### El programa se cierra inmediatamente

- Verifica los logs en stderr para ver el error específico
- Asegúrate de tener macOS 12.3+ (verifica con `sw_vers`)

## Desarrollo

### Estructura del Proyecto

```
swift-audio-capture/
├── Package.swift                    # Configuración del paquete Swift
├── Sources/
│   └── audio-capture/
│       ├── main.swift               # Punto de entrada del CLI
│       ├── AudioCapture.swift      # Lógica de ScreenCaptureKit
│       └── PCMConverter.swift      # Conversión a int16
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
