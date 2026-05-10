import { useMemo, useState } from "react";

const extensionFor = (name) => name.split(".").pop()?.toLowerCase() || "";

export default function useDocuments(addToast) {
  const [files, setFiles] = useState([]);

  const activeFiles = useMemo(
    () => files.filter((file) => file.status === "ready").map((file) => file.name),
    [files]
  );

  async function uploadFiles(fileList) {
    const incoming = Array.from(fileList || []);
    const supported = incoming.filter((file) => ["pdf", "docx", "txt"].includes(extensionFor(file.name)));
    const rejected = incoming.filter((file) => !["pdf", "docx", "txt"].includes(extensionFor(file.name)));

    rejected.forEach((file) => addToast("error", `${file.name} is not a supported file type.`));
    if (supported.length === 0) {
      return;
    }

    const uploadIds = supported.map((file) => ({
      id: `${file.name}-${file.lastModified}-${crypto.randomUUID()}`,
      name: file.name,
      type: extensionFor(file.name),
      status: "uploading",
      error: "",
    }));

    setFiles((current) => [...current, ...uploadIds]);

    const formData = new FormData();
    supported.forEach((file) => formData.append("files", file));

    window.setTimeout(() => {
      setFiles((current) =>
        current.map((file) =>
          uploadIds.some((upload) => upload.id === file.id) ? { ...file, status: "processing" } : file
        )
      );
    }, 250);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Upload failed");
      }

      setFiles((current) =>
        current.map((file) =>
          uploadIds.some((upload) => upload.id === file.id)
            ? {
                ...file,
                status: data.processed.includes(file.name) ? "ready" : "error",
                error: data.processed.includes(file.name) ? "" : "The server did not process this file.",
              }
            : file
        )
      );
      addToast("success", `${data.count} document${data.count === 1 ? "" : "s"} ready.`);
    } catch (error) {
      setFiles((current) =>
        current.map((file) =>
          uploadIds.some((upload) => upload.id === file.id)
            ? { ...file, status: "error", error: error.message }
            : file
        )
      );
      addToast("error", error.message || "Upload failed.");
    }
  }

  function removeFile(id) {
    setFiles((current) => current.filter((file) => file.id !== id));
  }

  return {
    files,
    activeFiles,
    uploadFiles,
    removeFile,
  };
}
