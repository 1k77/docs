(function () {
  "use strict";

  function getProjectFromPath(pathname) {
    // 兼容：/sources/<project>/... 以及 /_generated/sources/<project>/...
    var m = pathname.match(/\/(?:_generated\/)?sources\/([^\/]+)\//);
    return m ? m[1] : null;
  }

  function findSidebarContainer() {
    return document.querySelector(".wy-menu-vertical");
  }

  // 判断一个 href 是否“属于当前 project
  function hrefBelongsToProject(href, project) {
    if (!href) return false;
  
    // 外链直接排除
//    if (href.startsWith("http://") || href.startsWith("https://")) return false;

    // 仅在“当前页面已在 project 下”时，#anchor 才算 project
    if (href.startsWith("#")) {
      return /\/(?:_generated\/)?sources\/[^\/]+\//.test(window.location.pathname) &&
             (window.location.pathname.indexOf("/sources/" + project + "/") !== -1 ||
              window.location.pathname.indexOf("/_generated/sources/" + project + "/") !== -1);
    }

    try {
      var u = new URL(href, window.location.href);
      return (u.pathname.indexOf("/sources/" + project + "/") !== -1 ||
              u.pathname.indexOf("/_generated/sources/" + project + "/") !== -1);
    } catch (e) {
      return false;
    }
  }


  // 保留属于 project 的 li，并递归保留其父 li（防止断层）
  function scopeProjectLis(project, rootEl) {
    var lis = Array.prototype.slice.call(rootEl.querySelectorAll("li"));
    var keepSet = new Set();

    function markKeep(li) {
      if (!li || keepSet.has(li)) return;
      keepSet.add(li);
      var parentLi = li.parentElement ? li.parentElement.closest("li") : null;
      if (parentLi) markKeep(parentLi);
    }

    lis.forEach(function (li) {
      var links = Array.prototype.slice.call(li.querySelectorAll("a[href]"));
      var keep = links.some(function (a) {
        return hrefBelongsToProject(a.getAttribute("href"), project);
      });
      if (keep) markKeep(li);
    });

    lis.forEach(function (li) {
      li.style.display = keepSet.has(li) ? "" : "none";
    });

    // 高亮当前页面
    try {
      var cur = new URL(window.location.href).pathname.replace(/\/$/, "");
      var linksAll = rootEl.querySelectorAll("a[href]");
      linksAll.forEach(function (a) {
        a.classList.remove("ascend-current");
        var u = new URL(a.getAttribute("href"), window.location.href);
        if (u.pathname.replace(/\/$/, "") === cur) {
          a.classList.add("ascend-current");
          var li = a.closest("li");
          if (li) li.classList.add("current");
        }
      });
    } catch (e) {}
  }

  function findNextUlAfter(node) {
    var cur = node.nextElementSibling;
    while (cur) {
      if (cur.tagName && cur.tagName.toLowerCase() === "ul") return cur;
      cur = cur.nextElementSibling;
    }
    return null;
  }

  function scopeSidebarByCaption(project) {
    var menu = findSidebarContainer();
    if (!menu) return;

    var captions = Array.prototype.slice.call(menu.querySelectorAll("p.caption"));
    var anyKept = false;

    captions.forEach(function (cap) {
      var ul = findNextUlAfter(cap);
      if (!ul) return;

      var links = Array.prototype.slice.call(ul.querySelectorAll("a[href]"));
      var hasProject = links.some(function (a) {
        return hrefBelongsToProject(a.getAttribute("href"), project);
      });

      cap.style.display = hasProject ? "" : "none";
      ul.style.display = hasProject ? "" : "none";

      if (hasProject) {
        anyKept = true;
        scopeProjectLis(project, ul);
      }
    });

    // 兜底：防止意外清空
    if (!anyKept) {
      captions.forEach(function (cap) {
        cap.style.display = "";
        var ul = findNextUlAfter(cap);
        if (ul) ul.style.display = "";
      });
      scopeProjectLis(project, menu);
    }
  }

  function injectBackButton() {
    if (document.getElementById("ascend-back-to-portal")) return;

    var inner = document.querySelector(".wy-nav-content .rst-content");
    if (!inner) return;

    var div = document.createElement("div");
    div.id = "ascend-back-to-portal";
    div.style.margin = "12px 0 22px 0";

    var a = document.createElement("a");
    a.href = "../../index.html";
    a.textContent = "← 返回开源文档中心";
    a.style.display = "inline-block";
    a.style.padding = "8px 12px";
    a.style.border = "1px solid #e1e4e8";
    a.style.borderRadius = "8px";
    a.style.textDecoration = "none";

    div.appendChild(a);
    inner.insertBefore(div, inner.firstChild);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var project = getProjectFromPath(window.location.pathname || "");
    if (!project) return;

    scopeSidebarByCaption(project);
    injectBackButton();
  });
})();

