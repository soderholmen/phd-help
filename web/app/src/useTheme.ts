// Theme: data-theme on <html>, resolved BEFORE first paint by the
// inline script in index.html (no flash of the wrong theme). This hook
// owns everything after that: it follows the OS until the user
// toggles, then persists the choice and stops following. Persistence
// happens only in toggle — writing on mount would freeze the initial
// OS verdict and kill the following.
import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";

const KEY = "phd-theme";

function systemTheme(): Theme {
  return typeof matchMedia === "function" &&
    matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "dark" || v === "light" ? v : null;
  } catch {
    return null; // private mode: no persistence, still a theme
  }
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => stored() ?? systemTheme());

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (typeof matchMedia === "function") {
      const mq = matchMedia("(prefers-color-scheme: dark)");
      const onChange = (e: MediaQueryListEvent) => {
        if (stored()) return; // the user chose; their choice wins
        setTheme(e.matches ? "dark" : "light");
      };
      mq.addEventListener("change", onChange);
      return () => mq.removeEventListener("change", onChange);
    }
    return undefined;
  }, []);

  const toggle = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(KEY, next);
      } catch {
        /* no persistence available; the session still flips */
      }
      return next;
    });
  }, []);

  return [theme, toggle];
}
