# 🔍 Website Monitor

A simple website change monitoring service that sends email notifications when websites are updated. Built with Python, FastAPI, and PostgreSQL.

## ✨ Features

- **Smart Content Detection**: Filters out ads, timestamps, and dynamic content
- **Email Notifications**: Beautiful emails via Resend
- **User Dashboard**: Manage subscriptions easily
- **Security**: URL validation and input sanitization
- **Background Monitoring**: Automatic checks every 30 minutes

## 🚀 Quick Deploy to Railway

### Step 1: Get Your Resend API Key

1. Visit [resend.com](https://resend.com) and create an account
2. Add your domain (or use their test domain)
3. Generate an API key

### Step 2: Deploy to Railway

1. **Fork this repo** to your GitHub account

2. **Deploy to Railway**:
   - Visit [railway.app](https://railway.app) and sign up
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your forked repository

3. **Add PostgreSQL**:
   - In Railway, click "New Service" → "Database" → "PostgreSQL"

4. **Set Environment Variables**:
   ```
   RESEND_API_KEY=re_your_api_key_here
   FROM_EMAIL=Website Monitor <noreply@yourdomain.com>
   SECRET_KEY=generate-a-random-string-here
   BASE_URL=https://your-app-name.up.railway.app
   ```

5. **Deploy**: Railway auto-deploys when you push to main branch

That's it! Your website monitor is live in ~5 minutes. ⚡

## 📁 Simple Structure

```
website-monitor/
├── main.py              # Complete application (one file!)
├── requirements.txt     # Python dependencies
├── Procfile            # Railway configuration
├── templates/          # HTML templates
│   ├── index.html      # Homepage
│   └── dashboard.html  # User dashboard
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
└── README.md          # This file
```

## 🛠️ Local Development

1. **Clone and setup**:
   ```bash
   git clone your-repo-url
   cd website-monitor
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

3. **Setup database**:
   ```bash
   createdb website_monitor
   # Tables are created automatically on first run
   ```

4. **Run**:
   ```bash
   python main.py
   ```

Visit `http://localhost:8000`

## 🔧 How It Works

1. **User submits** email + URL through web form
2. **System takes** initial content snapshot  
3. **Background scheduler** checks every 30 minutes
4. **Smart filtering** removes dynamic content (ads, timestamps)
5. **Change detection** compares content hashes
6. **Email notification** sent when changes detected
7. **Dashboard** lets users manage subscriptions

## ⚙️ Configuration

All configuration through environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection | Yes (Railway provides) |
| `RESEND_API_KEY` | Email API key | Yes |
| `SECRET_KEY` | Random secret string | Yes |
| `BASE_URL` | Your app URL | Yes |
| `FROM_EMAIL` | Sender email address | Yes |

## 🐛 Troubleshooting

**Emails not sending?**
- Check RESEND_API_KEY is correct
- Verify FROM_EMAIL domain in Resend dashboard

**Database errors?**
- Railway provides DATABASE_URL automatically
- Check PostgreSQL service is running

**Scheduler not working?**
- Check Railway logs for errors
- Ensure RESEND_API_KEY is set

## 📈 Usage

1. Visit your Railway app URL
2. Enter email + website URL
3. Choose sensitivity level:
   - **Low**: Only major changes
   - **Medium**: Balanced detection  
   - **High**: Catches minor changes
4. Get email confirmation
5. Receive alerts when site changes!

## 🎯 Production Tips

- Use custom domain in Railway
- Monitor logs for any issues
- Keep dependencies updated
- Railway handles SSL & scaling automatically

## 📄 License

MIT License - use freely!

---

**Simple, effective website monitoring in one Python file.** 🚀