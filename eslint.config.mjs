import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

const eslintConfig = defineConfig([
  ...nextVitals,
  globalIgnores([
    ".next/**",
    ".astro/**",
    "dist/**",
    "public/pagefind/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
