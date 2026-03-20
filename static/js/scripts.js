// Audio Player State
const audio = document.getElementById('audioPlayer');
const mainPlayBtn = document.getElementById('mainPlayBtn');
const progressBarFill = document.getElementById('progressBarFill');
const currentTimeEl = document.getElementById('currentTime');
const durationEl = document.getElementById('duration');
const volBarFill = document.getElementById('volBarFill');

let currentPreviewUrl = null;
let _activeCardBtn = null; // currently highlighted card play button

// Default placeholder artwork
const defaultArt = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600"><rect width="600" height="600" fill="%231a0e36"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="200" fill="%23b388ff">🎵</text></svg>`;

// Audio Event Listeners
audio.addEventListener('timeupdate', () => {
  if (audio.duration) {
    const percent = (audio.currentTime / audio.duration) * 100;
    progressBarFill.style.width = percent + '%';
    currentTimeEl.textContent = formatTime(audio.currentTime);
  }
});

audio.addEventListener('loadedmetadata', () => {
  durationEl.textContent = formatTime(audio.duration);
});

audio.addEventListener('ended', () => {
  mainPlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
  progressBarFill.style.width = '0%';
  currentTimeEl.textContent = '0:00';
  _resetCardBtn();
});

function _resetCardBtn() {
  if (_activeCardBtn) {
    _activeCardBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    _activeCardBtn = null;
  }
}

function formatTime(seconds) {
  if (isNaN(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return Math.floor(m) + ':' + (s < 10 ? '0' : '') + Math.floor(s);
}

function togglePlay() {
  if (!audio.src || audio.src.includes(window.location.host) && audio.src.endsWith('/')) return;
  if (audio.paused) {
    audio.play();
    mainPlayBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
    if (_activeCardBtn) _activeCardBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
  } else {
    audio.pause();
    mainPlayBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    if (_activeCardBtn) _activeCardBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
  }
}

function playTrack(previewUrl, title, artist, artUrl, cardBtn = null) {
  if (!previewUrl) {
    showToast('No audio preview available for this track.', 'error');
    return;
  }

  // Reset any previously active card button
  _resetCardBtn();

  if (currentPreviewUrl === previewUrl) {
    // Same song — toggle play/pause
    togglePlay();
    // Re-attach btn since _resetCardBtn cleared it
    _activeCardBtn = cardBtn || null;
    return;
  }

  currentPreviewUrl = previewUrl;
  audio.src = previewUrl;

  document.getElementById('playerArt').src = artUrl || defaultArt;
  document.getElementById('playerArt').style.opacity = '1';
  document.getElementById('playerTitle').textContent = title;
  document.getElementById('playerArtist').textContent = artist;

  audio.play();
  mainPlayBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';

  // Update card button icon
  if (cardBtn) {
    _activeCardBtn = cardBtn;
    _activeCardBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
  }
}

function seek(e) {
  if (!audio.duration) return;
  const rect = document.getElementById('progressBarBg').getBoundingClientRect();
  const pos = (e.clientX - rect.left) / rect.width;
  audio.currentTime = pos * audio.duration;
}

function setVolume(e) {
  const rect = document.getElementById('volBarBg').getBoundingClientRect();
  const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  audio.volume = pos;
  volBarFill.style.width = (pos * 100) + '%';
}

audio.volume = 1;

async function fetchAlbumArt(songName, artistName) {
  try {
    const query = encodeURIComponent(songName + ' ' + artistName);
    const response = await fetch('https://itunes.apple.com/search?term=' + query + '&media=music&limit=1');
    const data = await response.json();
    if (data.results && data.results.length > 0) {
      return data.results[0].artworkUrl100.replace('100x100bb', '600x600bb');
    }
  } catch (e) {
    console.error('Art fetch error:', e);
  }
  return defaultArt;
}

// Auth State & UI
document.addEventListener('DOMContentLoaded', async () => {
  const authContainer = document.getElementById('authContainer');
  if (authContainer) {
    try {
      const res = await fetch('/api/auth/me');
      const data = await res.json();
      if (data.logged_in) {
        const initial = data.username[0].toUpperCase();
        authContainer.innerHTML = `
          <div style="position: relative;">
            <button onclick="toggleProfileMenu(event)" style="display: flex; align-items: center; gap: 10px; background: none; border: none; cursor: pointer; width: 100%; padding: 8px; border-radius: 8px; transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.05)'" onmouseout="this.style.background='none'">
              <div style="width: 34px; height: 34px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1rem; color: #fff; flex-shrink: 0;">${initial}</div>
              <span style="color: var(--text-main); font-weight: 600; font-size: 0.95rem;">${data.username}</span>
            </button>
            <div id="profileMenu" style="display: none; position: absolute; bottom: 48px; left: 0; right: 0; background: #1a0b36; border: 1px solid var(--border-color); border-radius: 10px; padding: 8px; z-index: 999; box-shadow: 0 8px 32px rgba(0,0,0,0.6);">
              <div style="padding: 8px 12px; margin-bottom: 4px; border-bottom: 1px solid var(--border-color);">
                <div style="font-weight: 700; color: var(--text-main);">${data.username}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Logged in</div>
              </div>
              <a href="/playlists" style="display: flex; align-items: center; gap: 10px; padding: 10px 12px; text-decoration: none; color: var(--text-muted); border-radius: 6px; transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.08)'" onmouseout="this.style.background='none'">
                <i class="fa-solid fa-heart"></i> Liked Songs
              </a>
              <button onclick="logout(event)" style="display: flex; align-items: center; gap: 10px; padding: 10px 12px; width: 100%; background: none; border: none; color: #ff8a80; cursor: pointer; font-size: inherit; font-family: inherit; border-radius: 6px; transition: background 0.2s;" onmouseover="this.style.background='rgba(255,82,82,0.1)'" onmouseout="this.style.background='none'">
                <i class="fa-solid fa-right-from-bracket"></i> Log Out
              </button>
            </div>
          </div>
        `;
      } else {
        authContainer.innerHTML = `<a href="/auth" class="nav-item"><i class="fa-solid fa-user"></i> Log In</a>`;
      }
    } catch (e) {
      authContainer.innerHTML = `<a href="/auth" class="nav-item"><i class="fa-solid fa-user"></i> Log In</a>`;
    }
  }

  // Global Search Logic
  const searchInput = document.getElementById('globalSearch');
  const searchResults = document.getElementById('searchResults');
  let searchTimeout = null;

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      clearTimeout(searchTimeout);
      const query = e.target.value.trim();

      if (query.length < 2) {
        searchResults.style.display = 'none';
        return;
      }

      searchTimeout = setTimeout(async () => {
        try {
          const res = await fetch('/api/search?q=' + encodeURIComponent(query));
          const data = await res.json();

          if (data.length === 0) {
            searchResults.innerHTML = '<div style="padding: 16px; color: var(--text-muted); text-align: center;">No results found</div>';
          } else {
            searchResults.innerHTML = data.map(item => `
                            <a href="/song/${item.track_id}" style="display: flex; align-items: center; gap: 12px; padding: 12px 16px; text-decoration: none; border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.2s;">
                                <div style="width: 40px; height: 40px; border-radius: 4px; background: rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; overflow: hidden;">
                                    <img src="${defaultArt}" alt="" id="search-img-${item.track_id}" style="width: 100%; height: 100%; object-fit: cover;">
                                </div>
                                <div style="overflow: hidden;">
                                    <div style="color: var(--text-main); font-weight: 600; font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.name}</div>
                                    <div style="color: var(--text-muted); font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.artist}</div>
                                </div>
                            </a>
                        `).join('');

            // Queue images
            data.forEach(item => {
              ImageQueue.add(async () => {
                const url = await fetchAlbumArt(item.name, item.artist);
                const img = document.getElementById(`search-img-${item.track_id}`);
                if (img) img.src = url;
              });
            });
          }
          searchResults.style.display = 'block';
        } catch (e) {
          searchResults.innerHTML = '<div style="padding: 16px; color: #ff5252; text-align: center;">Search failed</div>';
        }
      }, 300);
    });

    // Hide on outside click
    document.addEventListener('click', (e) => {
      if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
        searchResults.style.display = 'none';
      }
    });
  }
});

async function logout(e) {
  if (e) e.preventDefault();
  await fetch('/api/auth/logout', { method: 'POST' });
  window.location.reload();
}

async function addToPlaylist(songId, songName, artistName, previewUrl) {
  try {
    const res = await fetch('/api/playlist/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        song_id: songId,
        song_name: songName,
        artist_name: artistName,
        preview_url: previewUrl || ''
      })
    });

    if (res.status === 401) {
      showToast('Please log in to save songs!', 'error');
      setTimeout(() => window.location.href = '/auth', 1200);
      return;
    }

    const data = await res.json();
    showToast(data.message || 'Added to Liked Songs! ❤️', 'success');
  } catch (e) {
    showToast('Failed to connect to server.', 'error');
  }
}

function showToast(msg, type = 'success') {
  const existing = document.getElementById('toastMsg');
  if (existing) existing.remove();
  const t = document.createElement('div');
  t.id = 'toastMsg';
  t.textContent = msg;
  t.style.cssText = `position:fixed;bottom:110px;left:50%;transform:translateX(-50%) translateY(20px);background:${type === 'success' ? '#4caf50' : '#f44336'};color:#fff;padding:12px 24px;border-radius:24px;font-weight:600;font-size:0.9rem;z-index:9999;box-shadow:0 4px 20px rgba(0,0,0,0.5);transition:all 0.3s ease;opacity:0;`;
  document.body.appendChild(t);
  requestAnimationFrame(() => { t.style.opacity = '1'; t.style.transform = 'translateX(-50%) translateY(0)'; });
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 2500);
}

function toggleProfileMenu(e) {
  e.stopPropagation();
  const menu = document.getElementById('profileMenu');
  if (menu) menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}

document.addEventListener('click', () => {
  const menu = document.getElementById('profileMenu');
  if (menu) menu.style.display = 'none';
});

// Global Image Fetching Queue to prevent iTunes API rejection (429 Too Many Requests)
const ImageQueue = {
  queue: [],
  isProcessing: false,

  add: function (task) {
    this.queue.push(task);
    if (!this.isProcessing) this.process();
  },

  process: async function () {
    if (this.queue.length === 0) {
      this.isProcessing = false;
      return;
    }
    this.isProcessing = true;

    const task = this.queue.shift();
    try {
      await task();
    } catch (e) { }

    // Delay 150ms between requests
    setTimeout(() => this.process(), 200);
  }
};
