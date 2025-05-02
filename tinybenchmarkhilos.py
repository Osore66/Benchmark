import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import GPUtil
from tqdm import tqdm
import platform
import os

# Función para medir el uso de RAM
def get_ram_usage():
    return psutil.virtual_memory().percent

# Función para medir el uso de VRAM
def get_vram_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed / gpus[0].memoryTotal * 100
    return 0

# Función para medir el uso de carga de la GPU
def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].load * 100
    return 0

# Detectar GPU o CPU
if torch.cuda.is_available():
    device = "cuda"
    gpus = GPUtil.getGPUs()
    gpu_name = gpus[0].name if gpus else "Desconocida"
    print(f"✅ Se usará la GPU: {gpu_name}")
else:
    device = "cpu"
    cpu_name = platform.processor() or "Desconocido"
    torch.set_num_threads(os.cpu_count())  # 💪 Usa todos los hilos disponibles
    print(f"⚠️  No se ha detectado GPU. Se usará la CPU: {cpu_name}")
    print(f"🧵 Hilos de CPU: {torch.get_num_threads()}")

# Configuración estándar del benchmark
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Modificar con múltiplos de 2 el tamaño de batch_size para hacer más larga la prueba
batch_size = 2
num_batches = 50
max_new_tokens = 10
total_inferencias = batch_size * num_batches

# Texto simulado estandarizado
text_input = ["This is a sample input text for benchmarking."] * batch_size

# Cargar modelo y tokenizer
print("\nCargando modelo y tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)
model.eval()

# Tokenización única (evita cuello de botella CPU)
tokens = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
tokens = tokens.to(device)

# Warm-up
with torch.no_grad():
    _ = model.generate(**tokens, max_new_tokens=max_new_tokens)

# Benchmark
print(f"\n🚀 Ejecutando benchmark con {total_inferencias} inferencias usando {device.upper()}...\n")

total_tokens = 0
start = time.time()

# Barra de progreso
progress_bar = tqdm(
    range(num_batches),
    bar_format="{desc}",
)

for i in progress_bar:
    ram_after = get_ram_usage()

    if device == "cuda":
        vram_after = get_vram_usage()
        gpu_load = get_gpu_usage()
        custom_info = f"RAM={ram_after:.1f}%, VRAM={vram_after:.2f}%, GPU%={gpu_load:.1f}%"
    else:
        custom_info = f"RAM={ram_after:.1f}%"

    percentage = i / num_batches * 100
    remaining_time = (num_batches - i) * ((time.time() - start) / (i + 1)) if i > 0 else 0
    remaining_time_str = f"{int(remaining_time // 60):02d}:{int(remaining_time % 60):02d}"
    progress_bar.set_description_str(
        f"Progreso: {percentage:3.0f}% |{'█' * int(percentage / 10):<10}{' ' * (10 - int(percentage / 10))}| {remaining_time_str} | {custom_info}"
    )

    with torch.no_grad():
        output = model.generate(**tokens, max_new_tokens=max_new_tokens)

    total_tokens += output.shape[1]

if device == "cuda":
    torch.cuda.synchronize()

end = time.time()

# Resultados
print("\n📈 Resultados del benchmark:")
tiempo_total = end - start
inferencias_por_segundo = total_inferencias / tiempo_total
tokens_por_segundo = total_tokens / tiempo_total

print(f"🕒 Tiempo total (a menos, mejor): {tiempo_total:.4f} segundos")
print(f"⚡ Inferencias por segundo (a más, mejor): {inferencias_por_segundo:.2f}")
print(f"🔡 Tokens por segundo (a más, mejor): {tokens_por_segundo:.2f}")
# print(f"🔢 Tokens totales procesados: {total_tokens}")

