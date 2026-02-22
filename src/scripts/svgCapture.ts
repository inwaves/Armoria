const SVG_NS = "http://www.w3.org/2000/svg";
const XLINK_NS = "http://www.w3.org/1999/xlink";

const escapeSelector = (value: string): string => {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/([ !"#$%&'()*+,.\/:;<=>?@[\\\]^`{|}~])/g, "\\$1");
};

const getUseHref = (use: SVGUseElement): string | null =>
  use.getAttribute("href") ?? use.getAttributeNS(XLINK_NS, "href");

const ensureDefs = (svg: SVGSVGElement): SVGDefsElement => {
  let defs = svg.querySelector("defs");
  if (!defs) {
    defs = document.createElementNS(SVG_NS, "defs");
    svg.insertBefore(defs, svg.firstChild);
  }
  return defs;
};

const inlineUseElements = (svg: SVGSVGElement): void => {
  const defs = ensureDefs(svg);
  const seen = new Set<string>();
  let added = true;

  while (added) {
    added = false;
    const uses = Array.from(svg.querySelectorAll("use"));
    for (const use of uses) {
      const href = getUseHref(use);
      if (!href) continue;
      const hashIndex = href.indexOf("#");
      if (hashIndex == -1) continue;
      const id = href.slice(hashIndex + 1);
      if (!id || seen.has(id)) continue;

      const selector = `#${escapeSelector(id)}`;
      if (svg.querySelector(selector)) {
        seen.add(id);
        continue;
      }

      const referenced = document.getElementById(id);
      if (!referenced) continue;

      const clone = referenced.cloneNode(true) as Element;
      if (!clone.getAttribute("id")) {
        clone.setAttribute("id", id);
      }
      defs.appendChild(clone);
      seen.add(id);
      added = true;
    }
  }
};

export const coaSvgId = (id: string | number): string => `coa${id}`;

export const getSvgElementById = (id: string): SVGSVGElement | null => {
  const element = document.getElementById(id);
  if (!element) return null;
  if (element instanceof SVGSVGElement) return element;
  const svg = element.querySelector("svg");
  return svg instanceof SVGSVGElement ? svg : null;
};

export const captureSvgAsPng = async (
  svg: SVGSVGElement,
  size = 224
): Promise<string> => {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", SVG_NS);
  clone.setAttribute("xmlns:xlink", XLINK_NS);
  clone.setAttribute("width", `${size}`);
  clone.setAttribute("height", `${size}`);

  inlineUseElements(clone);

  const serialized = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([serialized], {type: "image/svg+xml;charset=utf-8"});
  const url = URL.createObjectURL(blob);

  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.crossOrigin = "anonymous";
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const context = canvas.getContext("2d");
      if (!context) {
        URL.revokeObjectURL(url);
        reject(new Error("Unable to acquire canvas context."));
        return;
      }
      context.clearRect(0, 0, size, size);
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(image, 0, 0, size, size);
      const dataUrl = canvas.toDataURL("image/png");
      URL.revokeObjectURL(url);
      resolve(dataUrl.replace(/^data:image\/png;base64,/, ""));
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to render SVG image."));
    };
    image.src = url;
  });
};

export const captureCoaAsPng = async (
  coaId: string | number,
  size = 224
): Promise<string | null> => {
  const svg = getSvgElementById(coaSvgId(coaId));
  if (!svg) return null;
  try {
    return await captureSvgAsPng(svg, size);
  } catch {
    return null;
  }
};

const nextFrame = () =>
  new Promise<void>(resolve => requestAnimationFrame(() => resolve()));

export const captureCoaBatch = async (
  ids: Array<string | number>,
  size = 224,
  batchSize = 25
): Promise<Array<string | null>> => {
  const results: Array<string | null> = [];
  for (let index = 0; index < ids.length; index += batchSize) {
    const batch = ids.slice(index, index + batchSize);
    const batchResults = await Promise.all(
      batch.map(id => captureCoaAsPng(id, size))
    );
    results.push(...batchResults);
    if (index + batchSize < ids.length) {
      await nextFrame();
    }
  }
  return results;
};