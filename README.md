# 🧠 Character-Level Language Generation with Neural Networks

## 📘 Overview

This project explores the ability of artificial intelligence to learn language at a very fine-grained level — character by character. The model can be trained on any piece of text and, once it's learned enough, it can generate entirely new content that resembles the original material.

It's designed to mimic patterns of human writing, learning not just words, but how letters form sequences that eventually create meaningful language. This foundational technique is what powers more complex AI systems seen in modern applications.

---

## 🎯 Objective

The goal is to build a lightweight yet effective language model capable of generating text based on learned patterns. Unlike models that understand full words, this system only sees individual letters and learns how to string them together over time.

This approach makes it:
- **Language-independent** – it can learn any alphabet or symbol.
- **Simple and elegant** – focused on understanding structure from scratch.
- **Educational** – ideal for demonstrating how neural networks grasp language structure.

---

## ⚙️ How It Works

1. **Data Loading**  
   The system begins by reading a literary dataset and identifying every unique character.

2. **Encoding and Mapping**  
   Each character is assigned a number. This way, the model can treat text like numerical data during learning.

3. **Model Training**  
   The model is taught to predict the next character based on a short sequence of previous ones. It gets feedback through its mistakes and gradually improves.

4. **Loss Evaluation**  
   To ensure the model is improving, its performance is tested regularly on unseen data.

5. **Text Generation**  
   Once trained, it can start with a single character and create an entirely new sequence that resembles the original style of writing.

---

## 🏗️ Model Details

The main system is based on a **Bigram Language Model**, meaning it focuses on learning relationships between character pairs (i.e., which character typically follows another). It’s a minimal yet surprisingly effective design for this purpose.

### Highlights:
- Learns from sequences of fixed size.
- Generates content without being told exact rules.
- Uses a form of memory to track previous characters while generating the next.

---

## 🧪 Sample Workflow

1. **Read and prepare the dataset**
2. **Split the data for training and evaluation**
3. **Train the model by making predictions and adjusting based on errors**
4. **Regularly check performance**
5. **Generate brand-new text using learned patterns**

---

## 🧰 Libraries Used

To bring this to life, the project makes use of the following tools:

| Library        | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `torch`        | Core library for building and training the learning model               |
| `torch.nn`     | Helps define the model’s structure and layers                           |
| `torch.nn.functional` | Provides functions used during training and generation          |

These libraries power the neural network, manage its learning process, and help with predictions.

---

## 🖨️ Output Example

After successful training, the model produces new text such as:

