# ⚡ EdgeReady

> Upload any neural network. Find out if it can run on embedded hardware — and how to fix it if it can't.

![EdgeReady Demo](assets/demo.gif)

---

## What is this?

Most embedded ML tools tell you numbers. EdgeReady tells you what those numbers **mean** and what to do about them.

You upload a trained model. EdgeReady runs it through a full deployment readiness check — parameter count, FLOP estimate, peak RAM usage, model size — and gives you a traffic light report against real embedded hardware targets.

If the model doesn't fit, you pick a compression technique, run it, and download the deployment-ready TFLite file.

Built by someone who has actually deployed models on STM32 and ESP32 hardware — and wanted a tool that explains what ST Edge AI Cloud shows you under the hood.

---

## Features

- **Deployment Readiness Report** — traffic light scoring across 4 key metrics with plain English explanations
- **Layer-by-Layer Breakdown** — interactive chart showing which layers consume the most compute
- **Compression Workshop** — three techniques with before/after comparison
  - INT8 Quantization
  - Pruning + Quantization  
  - Ternary Quantization (weights → -1, 0, +1)
- **Hardware Estimator** — theoretical flash, RAM, and latency estimates for 5 common chips
- **TFLite Export** — download your compressed model ready to flash

---

## Supported Formats

| Format | Extension |
|--------|-----------|
| Keras | `.h5`, `.keras` |
| TFLite | `.tflite` |
| ONNX | `.onnx` |

---

## Target Chips

| Chip | Flash | RAM | Notes |
|------|-------|-----|-------|
| STM32H7 (Cortex-M7) | 2MB | 1MB | High performance, used with Genx320 at IISc |
| STM32F4 (Cortex-M4) | 1MB | 192KB | Workhorse of embedded ML |
| ESP32-S3 (Xtensa LX7) | 8MB | 512KB | Vector extensions, good for IoT ML |
| ESP32 (Xtensa LX6) | 4MB | 520KB | No FPU — INT8 strongly recommended |
| Arduino Uno (ATmega328P) | 32KB | 2KB | Listed for reference — avoid ML here |

---

## Why not just use ST Edge AI Cloud?

ST's tool runs your model on actual hardware on their board farm. EdgeReady doesn't do that — it gives theoretical estimates based on chip specs.

What EdgeReady does that ST's tool doesn't:
- Works with any chip, not just STM32
- Accepts ONNX models, not just ST-specific formats
- Explains every number in plain English
- Lets you compress the model in the same workflow
- Is completely open source and free

---

## Run Locally

```bash
git clone https://github.com/purab-devlog/edgeready
cd edgeready
pip install -r requirements.txt
streamlit run app.py
```

## Run on Streamlit Cloud

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://edgeready.streamlit.app)

1. Fork this repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and select this repo
4. Set main file as `app.py`
5. Deploy

---

## Project Structure

```
edgeready/
├── app.py                        
├── core/
│   ├── loader.py                 
│   ├── analyser.py               
│   ├── compressor.py             
│   ├── hardware.py               
│   └── exporter.py               
├── ui/
│   ├── report_card.py            
│   ├── layer_viz.py              
│   ├── compression_ui.py         
│   └── hardware_ui.py            
├── data/
│   └── chip_specs.json           
├── assets/
│   └── demo.gif                  
├── requirements.txt
└── README.md
```

---

## Background

This project grew out of my time at IISc NeuRonICS Lab where I worked with STM32 H7 boards, Genx320 event cameras, and ESP32-S3 for embedded ML benchmarking. I used ST's Edge AI Developer Cloud regularly and wanted an open, educational version that explains what's happening under the hood — especially the compression techniques like ternary quantization that I researched as part of RISC-V performance benchmarking work.

---

## Tech Stack

`TensorFlow` `TFLite` `ONNX` `Streamlit` `Plotly` `tensorflow-model-optimization`

---

## License

MIT
