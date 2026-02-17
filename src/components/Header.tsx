"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { IconX, IconMenuDeep, IconSearch } from "./Icons";
import ThemeToggle from "./ThemeToggle";
import { SITE } from "@/config";

export default function Header() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  const currentPath =
    pathname.endsWith("/") && pathname !== "/"
      ? pathname.slice(0, -1)
      : pathname;

  const isActive = (path: string) => {
    const currentPathArray = currentPath.split("/").filter((p) => p.trim());
    const pathArray = path.split("/").filter((p) => p.trim());
    return currentPath === path || currentPathArray[0] === pathArray[0];
  };

  return (
    <>
      <a
        id="skip-to-content"
        href="#main-content"
        className="absolute start-16 -top-full z-50 bg-background px-3 py-2 text-accent backdrop-blur-lg transition-all focus:top-4"
      >
        Skip to content
      </a>

      <header
        className="sticky top-0 z-40 w-full border-b border-border/50 backdrop-blur-xl"
        style={{ background: "var(--header-bg, rgba(250,251,253,0.92))" }}
      >
        <div className="app-layout flex items-center justify-between py-3 sm:py-4">
          <Link
            href="/"
            className="text-xl font-semibold whitespace-nowrap text-accent sm:text-2xl"
          >
            {SITE.title}
          </Link>
          <nav id="nav-menu" className="flex items-center">
            <button
              className="focus-outline p-2 sm:hidden"
              aria-label={menuOpen ? "Close Menu" : "Open Menu"}
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen(!menuOpen)}
            >
              {menuOpen ? <IconX /> : <IconMenuDeep />}
            </button>
            <ul
              className={`${
                menuOpen ? "flex" : "hidden"
              } absolute top-full left-0 w-full flex-col border-b border-border/50 backdrop-blur-xl [&>li>a]:block [&>li>a]:px-6 [&>li>a]:py-3 [&>li>a]:font-medium [&>li>a]:hover:text-accent-gold sm:static sm:flex sm:w-auto sm:flex-row sm:border-none sm:bg-transparent sm:backdrop-blur-none sm:[&>li>a]:px-3 sm:[&>li>a]:py-1 sm:[&>li>a]:hover:text-accent-gold`}
              style={{
                background: menuOpen
                  ? "var(--menu-bg, rgba(250,251,253,0.95))"
                  : undefined,
              }}
            >
              <li>
                <Link
                  href="/"
                  className={currentPath === "/" ? "active-nav" : ""}
                  onClick={() => setMenuOpen(false)}
                >
                  Home
                </Link>
              </li>
              <li>
                <Link
                  href="/blog"
                  className={isActive("/blog") ? "active-nav" : ""}
                  onClick={() => setMenuOpen(false)}
                >
                  Blog
                </Link>
              </li>
              <li>
                <Link
                  href="/about"
                  className={isActive("/about") ? "active-nav" : ""}
                  onClick={() => setMenuOpen(false)}
                >
                  About
                </Link>
              </li>
              <li className="flex items-center justify-center px-3 sm:px-1">
                <Link
                  href="/search"
                  className={`focus-outline flex p-3 sm:p-1 ${isActive("/search") ? "[&>svg]:stroke-accent" : ""}`}
                  title="Search"
                  aria-label="search"
                  onClick={() => setMenuOpen(false)}
                >
                  <IconSearch />
                  <span className="sr-only">Search</span>
                </Link>
              </li>
              {SITE.lightAndDarkMode && (
                <li className="flex items-center justify-center px-3 sm:px-1">
                  <ThemeToggle />
                </li>
              )}
            </ul>
          </nav>
        </div>
      </header>
    </>
  );
}
