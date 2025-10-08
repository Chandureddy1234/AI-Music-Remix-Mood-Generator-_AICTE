# 🎵 AI Music Remix & Mood Generator# 🎵 AI Music Remix & Mood Generator



A powerful AI-powered music generation and remixing application built with Streamlit. Create original music, remix existing tracks, analyze mood, and unleash your creativity!Create, remix, and transform music using **100% FREE AI models**. No technical skills required - perfect for students!



## ✨ Features![License](https://img.shields.io/badge/license-MIT-blue.svg)

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

### 🎼 Music Generation![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)

- Generate original music from text prompts

- Multiple AI providers (Groq, Hugging Face, Free Generator)---

- Cloud-based generation (no local storage required)

- Customizable duration and style## 🎯 Overview



### 🎚️ Remix Engine**The Problem:**

- Upload and remix your audio filesStudents need expensive software and technical skills to create or remix music.

- Apply professional effects (reverb, delay, distortion, etc.)

- Real-time audio processing**The Solution:**

- Export high-quality remixed tracksA simple web app where anyone can:

- 🎵 Generate music from text ("upbeat dance music with synths")

### 🎭 Mood Analyzer- 🎚️ Remix songs with AI (separate vocals, change genre, add effects)

- AI-powered mood detection from audio- 🎯 Analyze mood and get creative suggestions

- Emotional analysis and visualization- 🎨 Layer and mix multiple tracks

- Genre and tempo detection

- Detailed audio insights**100% FREE** • **No Installation** • **No Music Skills Needed**



### 🎨 Creative Studio---

- Advanced audio manipulation tools

- Multiple effect chains## ✨ Key Features

- Professional-grade processing

- Real-time preview### 🎼 Music Generation

- Generate music from text descriptions

## 🚀 Quick Start- 15+ genres, 15+ moods

- Control BPM, duration (15-120s), and musical key

### Prerequisites- Batch generation for variations

- Python 3.8 or higher- Uses Meta's MusicGen (local or cloud)

- pip package manager

### 🎚️ Remix Engine  

### Installation- Upload audio files (MP3, WAV, FLAC, OGG)

- AI stem separation (vocals, drums, bass, instruments) with Demucs

1. Clone the repository:- Genre transfer (pop → lo-fi, rock → jazz)

```bash- Mood transformation

git clone https://github.com/Chandureddy1234/AI-Music-Remix-Mood-Generator-_AICTE.git- Tempo/pitch control

cd AI-Music-Remix-Mood-Generator-_AICTE- Audio effects (reverb, echo, distortion, lo-fi filter)

```

### 🎯 Mood Analyzer

2. Install dependencies:- Auto-detect mood and genre

```bash- Extract BPM, key, energy, danceability

pip install -r requirements.txt- Waveform and spectrogram visualizations

```- Get AI remix suggestions



3. Set up API keys (optional but recommended):### 🎨 Creative Studio

   - Create a `.streamlit` folder- Layer multiple tracks

   - Create `secrets.toml` file inside it- Real-time mixing

   - Add your API keys:- Volume controls per track

   ```toml- Export in multiple formats

   GROQ_API_KEY = "your_groq_api_key"

   HUGGINGFACE_TOKEN = "your_huggingface_token"### � History & More

   ```- Save favorites

- Search and filter generations

4. Run the application:- Mobile-responsive UI

```bash- Dark mode interface

streamlit run app.py

```---



## 🔑 API Keys (Optional)## 🆓 100% Free Technology



The app works with FREE cloud providers, but for better quality:| Component | Provider | Cost |

|-----------|----------|------|

- **Groq API**: Get free key at https://console.groq.com| Music Generation | Meta MusicGen | FREE |

- **Hugging Face**: Get free token at https://huggingface.co/settings/tokens| Stem Separation | Demucs | FREE |

| Cloud APIs | HuggingFace, Replicate | FREE tiers |

## 📦 Dependencies| LLM Enhancement | Groq (Llama 3.1) | FREE 30 req/min |

| Audio Effects | Pedalboard (Spotify) | FREE |

All dependencies are listed in `requirements.txt`:

- streamlit---

- torch

- torchaudio## 🚀 Quick Start

- pydub

- numpy### Prerequisites

- scipy- Python 3.9+

- requests- 4GB RAM (8GB recommended)

- groq- Internet connection

- huggingface-hub

- And more...### Installation (2 minutes)



## 🌐 Live Demo```bash

# 1. Clone repository

Deploy your own instance on Streamlit Cloud:git clone <your-repo-url>

cd "AI music generator"

1. Fork this repository

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)# 2. Create virtual environment

3. Sign in with GitHubpython -m venv venv

4. Click "New app"

5. Select this repository# Windows

6. Set main file: `app.py`venv\Scripts\activate

7. (Optional) Add API keys in Secrets

8. Deploy!# Mac/Linux

source venv/bin/activate

## 📁 Project Structure

# 3. Install dependencies

```pip install -r requirements.txt

├── app.py                      # Main Streamlit application

├── music_generator.py          # Music generation logic# 4. Run app

├── cloud_music_generator.py    # Cloud-based generationstreamlit run app.py

├── audio_processor.py          # Audio processing```

├── cloud_audio_analysis.py     # Cloud audio analysis

├── components.py               # UI componentsOpens at `http://localhost:8501`

├── config.py                   # Configuration

├── utils.py                    # Utility functions### Optional: Add API Keys (All FREE)

├── requirements.txt            # Python dependencies

├── style.css                   # Custom stylingGet a Groq API key for better prompts (30 req/min free):

└── utils/1. Visit https://console.groq.com

    ├── audio_utils.py          # Audio utilities2. Sign up and create API key

    └── file_utils.py           # File utilities3. Create `.env` file: `GROQ_API_KEY=gsk_your_key`

```

**Other free options:**

## 🎯 Usage- HuggingFace: https://huggingface.co/settings/tokens

- Replicate: https://replicate.com/account/api-tokens ($10 free credits)

### Generate Music

1. Navigate to "🎵 Music Generation"---

2. Enter a text prompt (e.g., "upbeat electronic dance music")

3. Select duration and provider## 📖 How to Use

4. Click "Generate Music"

5. Download your creation!### Generate Music

1. Go to **"Music Generation"** page

### Remix Audio2. Type: "chill lo-fi hip hop beat"

1. Navigate to "🎚️ Remix Engine"3. Select genre and mood

2. Upload an audio file (MP3, WAV, OGG)4. Click **"Generate"**

3. Apply effects and adjust parameters5. Download your track!

4. Preview and download remixed track

### Remix a Song

### Analyze Mood1. Go to **"Remix Engine"** page

1. Navigate to "🎭 Mood Analyzer"2. Upload audio file

2. Upload an audio file3. Choose: Separate stems / Change genre / Add effects

3. Get instant mood analysis4. Download remix

4. View detailed insights

### Analyze Music

## 🛠️ Technical Details1. Go to **"Mood Analyzer"** page  

2. Upload audio

- **Framework**: Streamlit3. View mood, BPM, energy, and visualizations

- **Audio Processing**: PyDub, Torch, Torchaudio

- **AI Models**: Groq, Hugging Face, Free Cloud Providers---

- **Supported Formats**: MP3, WAV, OGG, FLAC

- **Cloud-Based**: No local model storage required## 📁 Project Structure



## 🔒 Security```

AI music generator/

- API keys stored securely in Streamlit secrets├── app.py                 # Main application

- No audio files permanently stored├── music_generator.py     # AI music generation

- Temporary files cleaned automatically├── audio_processor.py     # Audio processing & effects

- .gitignore protects sensitive data├── components.py          # UI components

├── config.py              # Configuration

## 📝 License├── requirements.txt       # Dependencies

└── README.md             # This file

This project is open source and available for educational purposes.```



## 🤝 Contributing---



Contributions, issues, and feature requests are welcome!## 🎓 Perfect for Students



## 💬 Support✅ **No music theory needed** - Just describe what you want  

✅ **No expensive software** - Everything is free  

For issues or questions, please open an issue on GitHub.✅ **Quick results** - Generate in 30-60 seconds  

✅ **Learn by doing** - Experiment with genres and moods  

## 🎉 Credits✅ **School projects** - Create music for presentations  



Developed with ❤️ using Streamlit and various open-source AI models.---



---## 🔧 Troubleshooting



**Enjoy creating amazing music! 🎶****Models downloading slowly?**  

First run downloads models (~500MB), then cached locally.

**Out of memory?**  
Use cloud mode or smaller model size in settings.

**API key error?**  
Check `.env` file or add keys in Settings page.

---

## 🚀 Deploy to Streamlit Cloud (FREE Hosting)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "AI Music Generator"
git push origin main

# 2. Deploy
# Visit streamlit.io/cloud
# Connect GitHub repo
# Add API keys in Secrets
# Deploy!
```

Your app gets a free URL: `https://your-app.streamlit.app`

---

## 🤝 Contributing

Pull requests welcome! Please:
1. Fork the repo
2. Create feature branch
3. Test changes
4. Submit PR

---

## 📄 License

MIT License - Free to use, modify, and distribute!

---

## 🙏 Credits

- **Meta AI** - MusicGen & Demucs
- **Groq** - Fast LLM inference  
- **Spotify** - Pedalboard audio effects
- **Streamlit** - Web framework
- **HuggingFace** - Model hosting

---

## 📊 Stats

- **8,000+ lines of code**
- **25+ features**
- **15+ genres & moods**
- **100% FREE to use**

---

**Made with ❤️ for students and music lovers**

**⭐ Star this repo if you find it helpful!**
