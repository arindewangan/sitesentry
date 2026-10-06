/* SiteSentry supervisor queue - vanilla JS, no external dependencies. */
(function () {
  "use strict";

  var pendingList = document.getElementById("pendingList");
  var historyList = document.getElementById("historyList");
  var pendingCount = document.getElementById("pendingCount");
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

  function card(p, decided) {
    var d = document.createElement("div");
    d.className = "appr" + (decided ? " decided-" + p.status : "");
    var sev = String(p.severity || "high").toLowerCase();
    var html = "<h3>" + esc(p.title || p.action_id) +
      '<span class="sev sev-' + esc(sev) + '">' + esc(p.severity || "") + "</span></h3>" +
      "<p>" + esc(p.detail || "") + "</p>" +
      '<p class="muted">id: ' + esc(p.action_id) +
      " · raised: " + esc(p.created_at || "—") + "</p>";
    if (decided) {
      html += '<p>Decision: <strong>' + esc(p.status) + "</strong>" +
        (p.by ? " by " + esc(p.by) : "") +
        (p.decided_at ? " · " + esc(p.decided_at) : "") + "</p>";
    } else {
      html += '<div class="actions">' +
        '<button class="btn btn-primary" data-act="approve" data-id="' + esc(p.action_id) + '">Approve</button>' +
        '<button class="btn btn-danger" data-act="reject" data-id="' + esc(p.action_id) + '">Reject</button>' +
        "</div>";
    }
    d.innerHTML = html;
    return d;
  }

  function loadPending() {
    fetch("/api/pending").then(function (r) { return r.json(); }).then(function (j) {
      var items = j.pending || [];
      pendingCount.textContent = items.length;
      pendingList.innerHTML = "";
      if (!items.length) {
        pendingList.innerHTML = '<p class="muted">Queue empty — nothing awaiting approval.</p>';
        return;
      }
      items.forEach(function (p) { pendingList.appendChild(card(p, false)); });
    }).catch(function () { toast("Could not load pending approvals."); });
  }

  function loadHistory() {
    fetch("/api/history").then(function (r) { return r.json(); }).then(function (j) {
      var items = j.history || [];
      historyList.innerHTML = "";
      if (!items.length) {
        historyList.innerHTML = '<p class="muted">No decisions recorded yet.</p>';
        return;
      }
      items.slice(0, 20).forEach(function (p) { historyList.appendChild(card(p, true)); });
    }).catch(function () { toast("Could not load decision history."); });
  }

  function decide(actionId, verb) {
    fetch("/api/" + verb + "/" + encodeURIComponent(actionId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ by: "supervisor" })
    })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
      .then(function (res) {
        if (!res.ok || res.j.error) throw new Error((res.j && res.j.error) || "decision failed");
        toast("Action " + actionId + " " + res.j.decision + ".");
        loadPending();
        loadHistory();
      })
      .catch(function (e) { toast("Decision failed: " + e.message); });
  }

  pendingList.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-act]");
    if (!b) return;
    b.disabled = true;
    decide(b.getAttribute("data-id"), b.getAttribute("data-act"));
  });

  loadPending();
  loadHistory();
  setInterval(function () { loadPending(); loadHistory(); }, 5000);
})();
