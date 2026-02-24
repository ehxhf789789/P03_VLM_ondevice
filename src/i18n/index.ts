import { useState, useCallback } from "react";
import { ko } from "./ko";
import { en } from "./en";

export type Locale = "ko" | "en";
export type Translations = typeof ko;

const translations: Record<Locale, Translations> = { ko, en };

let currentLocale: Locale = "ko";
const listeners = new Set<() => void>();

function setLocale(locale: Locale) {
  currentLocale = locale;
  listeners.forEach((fn) => fn());
}

export function useTranslation() {
  const [, setTick] = useState(0);

  const forceUpdate = useCallback(() => setTick((t) => t + 1), []);

  // 컴포넌트 마운트 시 리스너 등록
  useState(() => {
    listeners.add(forceUpdate);
    return () => listeners.delete(forceUpdate);
  });

  return {
    t: translations[currentLocale],
    locale: currentLocale,
    setLocale,
  };
}
