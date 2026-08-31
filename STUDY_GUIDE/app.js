(function () {
  "use strict";

  var sidebar = document.getElementById("sidebar-nav");
  var main = document.getElementById("main-content");
  var order = CHAPTERS.map(function (c) { return c.id; });

  function findChapter(id) {
    for (var i = 0; i < CHAPTERS.length; i++) if (CHAPTERS[i].id === id) return CHAPTERS[i];
    return CHAPTERS[0];
  }

  function buildSidebar() {
    var frag = document.createDocumentFragment();

    var startBtn = document.createElement("button");
    startBtn.className = "start-here-btn";
    startBtn.textContent = "▶ Start Here";
    startBtn.addEventListener("click", function () { navigate(order[0]); });
    frag.appendChild(startBtn);

    CHAPTERS.forEach(function (ch) {
      var btn = document.createElement("button");
      btn.className = "nav-btn" + (ch.deep ? " deep-hide" : "");
      btn.dataset.id = ch.id;
      btn.innerHTML = '<span class="nav-icon">' + ch.icon + '</span><span>' + ch.nav + "</span>";
      btn.addEventListener("click", function () { navigate(ch.id); });
      frag.appendChild(btn);
    });

    var presiWrap = document.createElement("div");
    presiWrap.className = "presi-toggle-wrap";
    presiWrap.innerHTML =
      '<label class="presi-toggle"><input type="checkbox" id="presi-check"> Presentation Mode</label>' +
      '<div class="ptw-label">Hides the deep technical chapters and jumps straight to numbers, findings, status, Q&amp;A, and the presentation scripts.</div>';
    frag.appendChild(presiWrap);

    sidebar.appendChild(frag);

    document.getElementById("presi-check").addEventListener("change", function (e) {
      document.body.classList.toggle("presi-mode", e.target.checked);
      if (e.target.checked) {
        var cur = findChapter(getCurrentId());
        if (cur.deep) navigate("numbers");
      }
    });
  }

  var currentId = null;
  function getCurrentId() { return currentId || order[0]; }

  function renderQA(container) {
    container.innerHTML = "";
    QA.forEach(function (item, i) {
      var wrap = document.createElement("div");
      wrap.className = "qa-item";
      var qBtn = document.createElement("button");
      qBtn.className = "qa-q";
      qBtn.innerHTML = "<span>" + item.q + "</span><span class='qmark'>+</span>";
      var aDiv = document.createElement("div");
      aDiv.className = "qa-a";
      aDiv.innerHTML = "<p style='margin:0'>" + item.a + "</p>";
      qBtn.addEventListener("click", function () {
        wrap.classList.toggle("open");
        qBtn.querySelector(".qmark").textContent = wrap.classList.contains("open") ? "−" : "+";
      });
      wrap.appendChild(qBtn);
      wrap.appendChild(aDiv);
      container.appendChild(wrap);
    });
  }

  function navigate(id) {
    var ch = findChapter(id);
    currentId = ch.id;
    var idx = order.indexOf(ch.id);

    // sidebar active state
    var btns = sidebar.querySelectorAll(".nav-btn");
    btns.forEach(function (b) { b.classList.toggle("active", b.dataset.id === ch.id); });

    var prevId = order[idx - 1];
    var nextId = order[idx + 1];

    main.innerHTML =
      '<div class="progress-line">Chapter ' + (idx + 1) + " of " + order.length + "</div>" +
      '<h1 class="chapter-title">' + ch.title + "</h1>" +
      '<div class="chapter-body">' + ch.html + "</div>" +
      '<div class="chapter-nav-footer">' +
        '<button id="prev-btn"' + (prevId ? "" : " disabled") + '>← Previous</button>' +
        '<button id="next-btn"' + (nextId ? "" : " disabled") + '>Next →</button>' +
      "</div>";

    if (ch.id === "questions") {
      renderQA(document.getElementById("qa-list"));
    }

    var prevBtn = document.getElementById("prev-btn");
    var nextBtn = document.getElementById("next-btn");
    if (prevBtn) prevBtn.addEventListener("click", function () { navigate(prevId); });
    if (nextBtn) nextBtn.addEventListener("click", function () { navigate(nextId); });

    try { window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" }); } catch (e) { /* scrollTo unsupported in some hosts, non-fatal */ }
    try { history.replaceState(null, "", "#" + ch.id); } catch (e) { /* some hosts (e.g. sandboxed file:// contexts) restrict this, non-fatal */ }
  }

  buildSidebar();
  var initial = (location.hash || "").replace("#", "");
  navigate(order.indexOf(initial) >= 0 ? initial : order[0]);
})();
