document.addEventListener("DOMContentLoaded", () => {
    // === Sidebar Toggle ===
    const toggleBtn = document.getElementById('toggleSidebar');
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('mainContent');

    if (toggleBtn && sidebar && mainContent) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
        });
    }

    // === Header Shrink on Scroll ===
    const header = document.getElementById("pageHeader");
    let lastScrollTop = 0;
    window.addEventListener("scroll", function () {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        if (scrollTop > lastScrollTop && scrollTop > 50) {
            header?.classList.add("shrink");
        } else {
            header?.classList.remove("shrink");
        }
        lastScrollTop = Math.max(0, scrollTop);
    }, false);

    // === Fade-In Elements ===
    const fadeEls = document.querySelectorAll('.fade-in');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            } else {
                entry.target.classList.remove('visible');
            }
        });
    }, { threshold: 0.15 });
    fadeEls.forEach(el => observer.observe(el));

    // === Carousel Auto Scroll ===
    const track = document.getElementById('carousel');
    if (track) {
        const images = track.querySelectorAll('img');
        let index = 0;

        function updateCarousel() {
            images.forEach((img, i) => img.classList.toggle('active', i === index));

            const imageWidth = images[0].offsetWidth + 20;
            const trackWidth = track.offsetWidth;
            const centerOffset = (images[index].offsetLeft + imageWidth / 2) - (trackWidth / 2);

            track.scrollTo({ left: centerOffset, behavior: 'smooth' });
            index = (index + 1) % images.length;
        }

        setInterval(updateCarousel, 2000);
    }

    // === Download List Animation ===
    const entries = document.querySelectorAll(".file-entry");
    entries.forEach((el, i) => {
        el.style.opacity = 0;
        el.style.transform = "translateY(10px)";
        setTimeout(() => {
            el.style.transition = "all 0.5s ease";
            el.style.opacity = 1;
            el.style.transform = "translateY(0)";
        }, i * 100);
    });
});

window.onload = function () {
  if (typeof simplemaps_usmap !== "undefined") {
    simplemaps_usmap.hooks.state_mouseover = function (state_id) {
      const name = simplemaps_usmap_mapdata.state_specific[state_id].name;
      document.getElementById("state-hover-display").innerText = `📍 ${name} (${state_id})`;
    };

    simplemaps_usmap.hooks.state_mouseout = function () {
      document.getElementById("state-hover-display").innerText = "🗺️ Hover over a state to begin";
    };
  }
};

