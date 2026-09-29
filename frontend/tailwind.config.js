/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sigdex: {
          bg: "#0A0A0B",
          panel: "#141416",
          "panel-alt": "#1A1A1E",
          border: "#2A2A2E",
          "border-hover": "#3E3E44",
          amber: "#F5A623",
          "amber-hover": "#FFB84D",
          "amber-dark": "#D48B12",
          "amber-dim": "rgba(245, 166, 35, 0.12)",
          pink: "#E85D75",
          green: "#50E3C2",
          text: "#FFFFFF",
          muted: "#8E8E93",
          dim: "#555558",
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'Consolas', '"Courier New"', 'monospace'],
        sans: ['Inter', '"Segoe UI"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
