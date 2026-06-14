import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  statSync,
  watch
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = resolve(root, "dist");
const site = resolve(root, "site");
const pub = resolve(root, "public");
const shouldWatch = process.argv.includes("--watch");

function build() {
  rmSync(dist, { recursive: true, force: true });
  mkdirSync(dist, { recursive: true });
  copyDir(site, dist);
  if (existsSync(pub)) {
    copyDir(pub, dist);
  }
  console.log(`Built static site -> ${dist}`);
}

function copyDir(from, to) {
  mkdirSync(to, { recursive: true });
  for (const entry of readdirSync(from, { withFileTypes: true })) {
    const source = join(from, entry.name);
    const target = join(to, entry.name);
    if (entry.isDirectory()) {
      copyDir(source, target);
    } else if (entry.isFile() || statSync(source).isFile()) {
      copyFileSync(source, target);
    }
  }
}

build();

if (shouldWatch) {
  console.log("Watching site/ and public/ for changes...");
  for (const path of [site, pub]) {
    if (existsSync(path)) {
      watch(path, { recursive: true }, () => build());
    }
  }
}
