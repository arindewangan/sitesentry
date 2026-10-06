/* SiteSentry dashboard - vanilla JS, no external dependencies. */
(function () {
  "use strict";

  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("fileInput");
  var statusEl = document.getElementById("uploadStatus");
  var annotatedImg = document.getElementById("annotated");
  var noResult = document.getElementById("noResult");
  var latencyEl = document.getElementById("latency");
  var frameCountEl = document.getElementById("frameCount");
  var chipsEl = document.getElementById("eventChips");
  var traceEl = document.getElementById("trace");
  var pendingSnippet = document.getElementById("pendingSnippet");
  var evidenceEl = document.getElementById("evidence");
  var blurToggle = document.getElementById("blurToggle");
  var toastsEl = document.getElementById("toasts");

  function toast(msg) {
    var d = document.createElement("div");
    d.className = "toast";
    d.textContent = msg;
    toastsEl.appendChild(d);
    setTimeout(function () { d.remove(); }, 6000);
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function setStatus(msg) { statusEl.textContent = msg || ""; }

  function uploadFile(file) {
    if (!file) return;
    setStatus("Analyzing " + file.name + " …");
    var fd = new FormData();
    fd.append("file", file);
    fetch("/api/upload", { method: "POST", body: fd })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok) { throw new Error((res.j && res.j.error) || "upload failed"); }
        renderResult(res.j);
        setStatus("Done — " + (res.j.events || []).length + " event(s), " +
                  res.j.frames_processed + " frame(s), " +
                  res.j.per_frame_ms + " ms/frame.");
        refreshAll();
      })
      .catch(function (e) {
        setStatus("");
        toast("Upload failed: " + e.message);
      });
  }

  function renderResult(j) {
    if (j.annotated_image) {
      annotatedImg.src = "data:image/jpeg;base64," + j.annotated_image;
      annotatedImg.style.display = "block";
      noResult.style.display = "none";
    }
    latencyEl.textContent = j.per_frame_ms + " ms";
    frameCountEl.textContent = j.frames_processed;
    chipsEl.innerHTML = "";
    (j.events || []).forEach(function (ev) {
      var s = document.createElement("span");
      s.className = "chip sev-" + String(ev.severity || "low").toLowerCase();
      s.textContent = ev.type + " · " + ev.severity;
      chipsEl.appendChild(s);
    });
    if (!(j.events || []).length) {
      chipsEl.innerHTML = '<span class="chip">no incidents detected</span>';
    }
  }

  /* --- upload wiring --- */
  dropzone.addEventListener("click", function () { fileInput.click(); });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { fileInput.click(); }
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
    fileInput.value = "";
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.remove("dragover"); });
  });
  dropzone.addEventListener("drop", function (e) {
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });

  /* --- synthetic sample buttons --- */
  document.querySelectorAll(".sample-btn").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var n = btn.getAttribute("data-sample");
      setStatus("Loading synthetic sample " + n + " …");
      fetch("/api/sample/" + n)
        .then(function (r) {
          if (!r.ok) throw new Error("sample " + n + " unavailable");
          return r.blob();
        })
        .then(function (blob) {
          uploadFile(new File([blob], "sample" + n + ".jpg", { type: "image/jpeg" }));
        })
        .catch(function (err) { setStatus(""); toast(err.message); });
    });
  });

  /* --- webcam note --- */
  document.getElementById("streamBtn").addEventListener("click", function () {
    fetch("/api/stream")
      .then(function (r) {
        if (r.status === 503) {
          return r.json().then(function (j) {
            toast("Live stream unavailable: " + (j.hint || j.error) + ". Use upload or samples.");
          });
        }
        toast("Live stream endpoint is reachable — open /api/stream in a new tab.");
      })
      .catch(function () { toast("Could not reach the stream endpoint."); });
  });

  /* --- face-blur toggle --- */
  blurToggle.addEventListener("change", function () {
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blur_faces: blurToggle.checked })
    })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (!j.ok) throw new Error("config rejected");
        toast("Face blur " + (blurToggle.checked ? "enabled" : "disabled") + ".");
      })
      .catch(function () { toast("Could not update face-blur setting."); });
  });

  /* --- polling --- */
  function pollTrace() {
    fetch("/api/trace").then(function (r) { return r.json(); }).then(function (j) {
      var items = (j.trace || []).slice(-14).reverse();
      traceEl.innerHTML = "";
      if (!items.length) {
        traceEl.innerHTML = '<li class="step-perceive"><span class="t-step">idle</span>No agent activity yet.</li>';
        return;
      }
      items.forEach(function (t) {
        var li = document.createElement("li");
        var step = String(t.step || "act").toLowerCase();
        li.className = "step-" + (["perceive", "plan", "act"].indexOf(step) >= 0 ? step : "act");
        var time = "";
        try { time = new Date(t.t).toLocaleTimeString(); } catch (e) { /* ignore */ }
        li.innerHTML = '<span class="t-step">' + esc(step) + "</span>" +
                       esc(t.text || "") +
                       '<span class="t-time">' + esc(time) + "</span>";
        traceEl.appendChild(li);
      });
    }).catch(function () { /* keep last rendered */ });
  }

  function pollPending() {
    fetch("/api/pending").then(function (r) { return r.json(); }).then(function (j) {
      var items = j.pending || [];
      if (!items.length) {
        pendingSnippet.innerHTML = '<p class="muted">Queue empty — nothing awaiting approval.</p>';
        return;
      }
      pendingSnippet.innerHTML = "";
      items.slice(0, 4).forEach(function (p) {
        var d = document.createElement("div");
        d.className = "appr";
        d.innerHTML = "<h3>" + esc(p.title || p.action_id) +
          '<span class="sev sev-' + esc(String(p.severity || "high").toLowerCase()) + '">' +
          esc(p.severity || "") + "</span></h3>" +
          "<p>" + esc(p.detail || "") + "</p>" +
          '<p class="muted">' + esc(p.action_id) + "</p>";
        pendingSnippet.appendChild(d);
      });
      var more = document.createElement("p");
      more.className = "muted";
      more.innerHTML = items.length > 4
        ? "…and " + (items.length - 4) + " more — <a href='/supervisor'>open supervisor queue</a>"
        : "<a href='/supervisor'>Open supervisor queue</a>";
      pendingSnippet.appendChild(more);
    }).catch(function () { /* keep last rendered */ });
  }

  function pollEvents() {
    fetch("/api/events").then(function (r) { return r.json(); }).then(function (j) {
      var evs = j.events || [];
      evidenceEl.innerHTML = "";
      var shown = 0;
      evs.forEach(function (ev) {
        if (!ev.evidence || shown >= 12) return;
        var img = document.createElement("img");
        img.src = "data:image/jpeg;base64," + ev.evidence;
        img.alt = esc(ev.type || "evidence");
        img.title = esc((ev.type || "") + " · " + (ev.severity || ""));
        evidenceEl.appendChild(img);
        shown++;
      });
      if (!shown) evidenceEl.innerHTML = '<p class="muted">No evidence yet.</p>';
    }).catch(function () { /* keep last rendered */ });
  }

  function pollSummary() {
    fetch("/api/summary").then(function (r) { return r.json(); }).then(function (j) {
      document.getElementById("sumTotal").textContent = j.total_incidents;
      document.getElementById("sumAvg").textContent = j.avg_latency_ms + " ms";
      document.getElementById("sumFrames").textContent = j.frames_processed;
      var sev = document.getElementById("sevBadges");
      sev.innerHTML = "";
      Object.keys(j.by_severity || {}).sort().forEach(function (k) {
        var s = document.createElement("span");
        s.className = "sbadge";
        s.textContent = k + ": " + j.by_severity[k];
        sev.appendChild(s);
      });
      var typ = document.getElementById("typeBadges");
      typ.innerHTML = "";
      Object.keys(j.by_type || {}).sort().forEach(function (k) {
        var s = document.createElement("span");
        s.className = "sbadge";
        s.textContent = k + ": " + j.by_type[k];
        typ.appendChild(s);
      });
    }).catch(function () { /* keep last rendered */ });
  }

  function pollAgreement() {
    fetch("/api/agreement").then(function (r) { return r.json(); }).then(function (j) {
      var el = document.getElementById("agreement");
      var rate = j.agreement_rate == null ? "n/a" : (j.agreement_rate * 100).toFixed(1) + "%";
      el.innerHTML =
        '<p>Agreement rate: <span class="big">' + esc(rate) + "</span></p>" +
        "<p>agreed: <strong>" + j.agreed + "</strong> · disagreed: <strong>" + j.disagreed +
        "</strong> · total compared: <strong>" + j.total + "</strong></p>" +
        '<p class="muted">' + esc(j.note || "") + "</p>";
    }).catch(function () { /* keep last rendered */ });
  }

  function refreshAll() { pollTrace(); pollPending(); pollEvents(); pollSummary(); pollAgreement(); }

  setInterval(pollTrace, 2000);
  setInterval(function () { pollPending(); pollEvents(); pollSummary(); pollAgreement(); }, 3000);
  refreshAll();
})();
