import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          500: "#00C2A8",
          600: "#009F8A"
        }
      },
      boxShadow: {
        soft: "0 10px 30px rgba(2, 6, 23, 0.12)"
      }
    }
  },
  plugins: []
};

export default config;
