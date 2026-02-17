import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import { API_KEY } from "../constants";

const blobToDataUrl = (blob: Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result;
      if (typeof result === "string") {
        resolve(result);
      } else {
        reject(new Error("Failed to read image blob."));
      }
    };
    reader.onerror = () => reject(new Error("Failed to read image blob."));
    reader.readAsDataURL(blob);
  });

const inlineImages = async (element: HTMLElement) => {
  const imgs = Array.from(element.querySelectorAll("img"));
  const originals: Array<{ img: HTMLImageElement; src: string }> = [];

  await Promise.all(
    imgs.map(async (img) => {
      const src = img.currentSrc || img.src;
      if (!src || src.startsWith("data:")) return;
      try {
        const headers: Record<string, string> = {};
        if (API_KEY) headers["x-api-key"] = API_KEY;
        const response = await fetch(src, { headers });
        if (!response.ok) return;
        const blob = await response.blob();
        const dataUrl = await blobToDataUrl(blob);
        originals.push({ img, src: img.src });
        img.src = dataUrl;
      } catch {
        // keep original src on failure
      }
    })
  );

  return () => {
    originals.forEach(({ img, src }) => {
      img.src = src;
    });
  };
};

const waitForImageDecode = async (element: HTMLElement) => {
  const imgs = Array.from(element.querySelectorAll("img"));
  await Promise.all(
    imgs.map(async (img) => {
      try {
        if (!img.complete) {
          await new Promise<void>((resolve) => {
            const done = () => resolve();
            img.addEventListener("load", done, { once: true });
            img.addEventListener("error", done, { once: true });
          });
        }
        if ("decode" in img) {
          await img.decode().catch(() => undefined);
        }
      } catch {
        // do not block PDF generation due to a single image
      }
    })
  );
};

const calculateSafeScale = (element: HTMLElement) => {
  const width = Math.max(element.scrollWidth, element.clientWidth, 1);
  const height = Math.max(element.scrollHeight, element.clientHeight, 1);
  const maxSideScale = 16384 / Math.max(width, height);
  const maxAreaScale = Math.sqrt(268435456 / (width * height));
  const safeScale = Math.min(2, maxSideScale, maxAreaScale);
  return Number.isFinite(safeScale) && safeScale > 0 ? safeScale : 1;
};

const normalizeUnsupportedColors = (doc: Document) => {
  const replaceColorFn = (value: string) =>
    value.replace(/oklch\([^)]+\)/gi, "rgb(107, 114, 128)");

  const styles = Array.from(doc.querySelectorAll("style"));
  styles.forEach((styleTag) => {
    if (!styleTag.textContent || !/oklch\(/i.test(styleTag.textContent)) return;
    styleTag.textContent = replaceColorFn(styleTag.textContent);
  });

  const inlineStyled = Array.from(doc.querySelectorAll<HTMLElement>("[style]"));
  inlineStyled.forEach((node) => {
    const styleText = node.getAttribute("style");
    if (!styleText || !/oklch\(/i.test(styleText)) return;
    node.setAttribute("style", replaceColorFn(styleText));
  });
};

const expandScrollableContent = (doc: Document) => {
  const selectors = [
    "[data-pdf-expand]",
    ".overflow-auto",
    ".overflow-y-auto",
    ".overflow-x-auto",
    "pre",
  ];
  const nodes = Array.from(
    doc.querySelectorAll<HTMLElement>(selectors.join(","))
  );

  nodes.forEach((node) => {
    node.style.maxHeight = "none";
    node.style.height = "auto";
    node.style.overflow = "visible";
  });
};

export const exportElementToPdf = async (
  element: HTMLElement,
  filename: string
) => {
  const restoreImages = await inlineImages(element);
  await waitForImageDecode(element);

  const scale = calculateSafeScale(element);
  let canvas: HTMLCanvasElement;

  try {
    try {
      canvas = await html2canvas(element, {
        scale,
        useCORS: true,
        backgroundColor: "#ffffff",
        logging: false,
        imageTimeout: 15000,
        scrollX: 0,
        scrollY: -window.scrollY,
        onclone: (clonedDoc) => {
          normalizeUnsupportedColors(clonedDoc);
          expandScrollableContent(clonedDoc);
        },
      });
    } catch {
      // Fallback: retry with lower scale if the first render exceeded limits.
      canvas = await html2canvas(element, {
        scale: 1,
        useCORS: true,
        backgroundColor: "#ffffff",
        logging: false,
        imageTimeout: 15000,
        scrollX: 0,
        scrollY: -window.scrollY,
        onclone: (clonedDoc) => {
          normalizeUnsupportedColors(clonedDoc);
          expandScrollableContent(clonedDoc);
        },
      });
    }

    const pdf = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    });

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 8;
    const contentWidth = pageWidth - margin * 2;
    const drawableHeight = pageHeight - margin * 2;
    const pageHeightPx = Math.floor((drawableHeight * canvas.width) / contentWidth);

    let renderedHeightPx = 0;
    let pageIndex = 0;

    while (renderedHeightPx < canvas.height) {
      const sliceHeightPx = Math.min(pageHeightPx, canvas.height - renderedHeightPx);
      const pageCanvas = document.createElement("canvas");
      pageCanvas.width = canvas.width;
      pageCanvas.height = sliceHeightPx;
      const ctx = pageCanvas.getContext("2d");
      if (!ctx) {
        throw new Error("Could not build PDF page canvas.");
      }
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, pageCanvas.width, pageCanvas.height);
      ctx.drawImage(
        canvas,
        0,
        renderedHeightPx,
        canvas.width,
        sliceHeightPx,
        0,
        0,
        pageCanvas.width,
        sliceHeightPx
      );

      const pageImage = pageCanvas.toDataURL("image/png");
      const renderedHeightMm = (sliceHeightPx * contentWidth) / canvas.width;

      if (pageIndex > 0) {
        pdf.addPage();
      }
      pdf.addImage(pageImage, "PNG", margin, margin, contentWidth, renderedHeightMm);

      renderedHeightPx += sliceHeightPx;
      pageIndex += 1;
    }

    const appendixNodes = Array.from(
      element.querySelectorAll<HTMLElement>("[data-pdf-append-text]")
    );
    const appendixText = appendixNodes
      .map((node) => (node.textContent || "").trim())
      .filter(Boolean)
      .join("\n\n");

    if (appendixText) {
      const maxWidth = pageWidth - margin * 2;
      const lineHeight = 4.5;
      const startY = margin + 8;
      const pageBottom = pageHeight - margin;
      let y = startY;

      pdf.addPage();
      pdf.setFont("courier", "normal");
      pdf.setFontSize(9);
      pdf.text("Metadata Appendix", margin, margin);

      const lines = pdf.splitTextToSize(appendixText, maxWidth);
      for (const line of lines) {
        if (y > pageBottom) {
          pdf.addPage();
          pdf.setFont("courier", "normal");
          pdf.setFontSize(9);
          y = startY;
        }
        pdf.text(line, margin, y);
        y += lineHeight;
      }
    }

    pdf.save(filename);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    throw new Error(`PDF export failed: ${message}`);
  } finally {
    restoreImages();
  }
};
