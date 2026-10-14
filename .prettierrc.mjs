/** @type {import("prettier").Config} */
const config = {
  arrowParens: "avoid",
  semi: true,
  tabWidth: 2,
  printWidth: 80,
  singleQuote: false,
  jsxSingleQuote: false,
  trailingComma: "es5",
  bracketSpacing: true,
  endOfLine: "lf",
  plugins: ["prettier-plugin-tailwindcss"],
  tailwindStylesheet: "./src/styles/global.css",
};

export default config;
