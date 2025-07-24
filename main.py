import cv2
import torch
import time
import threading
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
# model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
# model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to("cuda")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not access webcam")

INTERVAL_MS = 1000
last_processed_time = 0
last_text = ""
processing_lock = threading.Lock()
is_processing = False

def process_frame(frame):
    global last_text, is_processing
    with processing_lock:
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": "Describe this image in maximum 5 words."},
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            if "Assistant:" in generated_texts[0]:
                generated_texts[0] = generated_texts[0].split("Assistant:")[-1].strip()
            last_text = generated_texts[0]
        finally:
            is_processing = False

print("Webcam running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time() * 1000
        if not is_processing and current_time - last_processed_time >= INTERVAL_MS:
            last_processed_time = current_time
            is_processing = True
            threading.Thread(target=process_frame, args=(frame.copy(),), daemon=True).start()

        display_frame = frame.copy()

        if last_text:
            height, width, _ = display_frame.shape
            strip_height = 40
            cv2.rectangle(display_frame, (0, height - strip_height), (width, height), (0, 0, 0), -1)
            cv2.putText(display_frame, last_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("SmolVLM2 Webcam Demo", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
