function initFadeIn() {
  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const el = entry.target as HTMLElement;
          el.classList.add("is-visible");

          // Stagger children if this is a stagger container
          if (el.classList.contains("fade-in-stagger")) {
            const children = Array.from(el.children) as HTMLElement[];
            children.forEach((child, i) => {
              child.style.transitionDelay = `${i * 80}ms`;
            });
          }

          observer.unobserve(el);
        }
      });
    },
    {
      threshold: 0.1,
      rootMargin: "0px 0px -40px 0px",
    }
  );

  document.querySelectorAll(".fade-in, .fade-in-stagger").forEach(el => {
    observer.observe(el);
  });
}

// Run on initial load
initFadeIn();

// Run on Astro page transitions
document.addEventListener("astro:page-load", initFadeIn);
