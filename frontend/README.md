# Frontend

React/Vite frontend for Fírinne Dhigiteach.

## Purpose

The frontend provides:

- upload flow for image, video, and audio files
- case history and filtering
- image, video, and audio analysis detail pages
- print/report views
- settings page for runtime configuration
- audit log view

## Development

Install dependencies:

```powershell
npm install
```

Run the development server:

```powershell
npm run dev
```

Default URL:

```text
http://localhost:5173
```

## Build

```powershell
npm run build
```

This runs TypeScript project checks and creates a Vite production build.

## Environment

Optional `frontend/.env` values:

```text
VITE_API_BASE_URL=http://localhost:8000
VITE_API_KEY=
VITE_ADMIN_KEY=
```

`VITE_ADMIN_KEY` must match the backend admin key if settings updates are made through the UI.
