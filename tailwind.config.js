/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        grade: {
          S: "#6366f1",
          A: "#22c55e",
          B: "#eab308",
          C: "#f97316",
          D: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
