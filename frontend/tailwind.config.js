/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#060B14',
          900: '#0B1120',
          850: '#0F172A',
          800: '#15203B',
          750: '#1E293B',
          700: '#334155',
        },
        cyan: {
          400: '#22D3EE',
          500: '#06B6D4',
        },
        brand: {
          blue: '#0284C7',
          lightBlue: '#38BDF8',
          accent: '#0EA5E9',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
