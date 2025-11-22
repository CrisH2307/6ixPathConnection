import { useEffect, useMemo, useState } from "react";

const STAGES = {
  UPLOAD: "upload",
  DETAILS: "details",
  PATH_LOADING: "path_loading",
  PATH_VIEW: "path_view",
};

const STAGE_FLOW = [
  { id: STAGES.UPLOAD, label: "Resume" },
  { id: STAGES.DETAILS, label: "Details" },
  { id: STAGES.PATH_LOADING, label: "Path build" },
  { id: STAGES.PATH_VIEW, label: "Paths" },
];

const SKILL_LIBRARY = ["Python", "TypeScript", "React", "APIs", "Cloud", "DevOps", "SQL", "Data Engineering"];
const KEYWORD_LIBRARY = ["new grad", "mentorship", "alumni", "hiring", "referrals", "technology", "coffee chat", "warm intro"];
const ROLE_LIBRARY = ["Software Engineer", "DevOps Engineer", "Data Engineer", "Backend Engineer", "SWE - Platform"];
const STATUS_STATES = ["Not started", "Contacted", "Replied"];

const API_BASE = process.env.REACT_APP_SERVICE_URL || "http://localhost:8000";

const AnimatedLoader = ({ title, subtitle, icon }) => (
  <div className="flex flex-col items-center gap-3 text-center">
    <div className="relative h-60 w-60">
      <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-indigo-500/30 via-sky-400/25 to-lime-400/20 blur-lg animate-pulse" />
      <div className="absolute inset-2 rounded-full border-2 border-indigo-200/80" />
      <div className="absolute inset-3 rounded-full border-2 border-dashed border-indigo-500/80 animate-spin" style={{ animationDuration: "8s" }} />
      <div className="absolute inset-6 rounded-full bg-white shadow-md" />
      <div className="absolute -left-1 top-5 h-3 w-3 rounded-full bg-sky-400 shadow animate-bounce" />
      <div className="absolute right-0 bottom-4 h-3 w-3 rounded-full bg-lime-400 shadow animate-ping" />
      <div className="absolute inset-8 flex items-center justify-center">{icon || <div className="h-4 w-4 rounded-full bg-indigo-600 shadow" />}</div>
    </div>
    <div className="space-y-1">
      <p className="text-sm font-semibold text-slate-900">{title}</p>
      {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
    </div>
  </div>
);

const formatTagLabel = (tag = "") =>
  tag
    .toString()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());

const buildSignalList = (signals = {}, connectionTags = []) => {
  if (!signals || typeof signals !== "object") return ["No obvious shared signals"];
  const rows = [];
  if (signals.same_school) rows.push("Same school");
  if (signals.same_company) rows.push("Same company");
  if (typeof signals.skill_similarity === "number") rows.push(`Skill similarity: ${signals.skill_similarity.toFixed(2)}`);
  if (typeof signals.relationship_strength === "number") rows.push(`Warm mutual strength: ${signals.relationship_strength.toFixed(2)}`);
  if (signals.categories_a && signals.categories_b) {
    const shared = signals.categories_a.filter((item) => signals.categories_b.includes(item));
    if (shared.length) rows.push(`Shared focus: ${shared.join(", ")}`);
  }
  if (Array.isArray(connectionTags) && connectionTags.length) {
    rows.push(`Personal link: ${connectionTags.map(formatTagLabel).join(", ")}`);
  }
  if (!rows.length) rows.push("No obvious shared signals");
  return rows;
};

const TagEditor = ({ label, values, onChange, suggestions }) => {
  const [open, setOpen] = useState(false);

  const available = useMemo(() => {
    const lower = values.map((v) => v.toLowerCase());
    return suggestions.filter((opt) => !lower.includes(opt.toLowerCase()));
  }, [values, suggestions]);

  const handleRemove = (index) => onChange(values.filter((_, i) => i !== index));
  const handleRename = (index, nextValue) => {
    const next = values.map((val, i) => (i === index ? nextValue : val));
    onChange(next);
  };
  const handleAdd = (value) => {
    if (!value) return;
    const exists = values.some((val) => val.toLowerCase() === value.toLowerCase());
    if (!exists) onChange([...values, value]);
  };

  return (
    <div className="space-y-3 rounded-3xl border border-slate-200 p-4">
      <div className="flex items-center justify-between text-sm font-semibold text-slate-600">
        <span>{label}</span>
        <button type="button" className="text-blue-600 underline-offset-4 hover:underline" onClick={() => setOpen((prev) => !prev)}>
          Add more?
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {values.length === 0 && <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">No {label.toLowerCase()}</span>}
        {values.map((value, index) => (
          <EditableTag key={`${value}-${index}`} value={value} onRemove={() => handleRemove(index)} onRename={(val) => handleRename(index, val)} />
        ))}
      </div>
      {open && (
        <div className="flex flex-wrap gap-2">
          {available.length === 0 ? (
            <span className="text-xs text-slate-500">All suggestions already added</span>
          ) : (
            available.map((option) => (
              <button key={option} type="button" className="rounded-full border px-3 py-1 text-xs text-slate-700 hover:bg-slate-50" onClick={() => handleAdd(option)}>
                {option}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
};

const EditableTag = ({ value, onRemove, onRename }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  useEffect(() => setDraft(value), [value]);

  const commit = () => {
    const next = draft.trim();
    if (!next) {
      onRemove();
      return;
    }
    onRename(next);
    setIsEditing(false);
  };

  return (
    <span className={`inline-flex items-center gap-2 rounded-full bg-indigo-100 px-3 py-1 text-xs font-medium text-slate-800 ${isEditing ? "ring-2 ring-indigo-300" : ""}`}>
      {isEditing ? (
        <input
          autoFocus
          className="w-28 bg-transparent text-sm outline-none"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") {
              setDraft(value);
              setIsEditing(false);
            }
          }}
        />
      ) : (
        <>
          <button type="button" className="text-left" onClick={() => setIsEditing(true)}>
            {value}
          </button>
          <button type="button" onClick={onRemove}>
            ×
          </button>
        </>
      )}
    </span>
  );
};

const StageHeading = ({ stage }) => {
  const position = STAGE_FLOW.findIndex((item) => item.id === stage);
  return (
    <div className="flex flex-wrap gap-2">
      {STAGE_FLOW.map((item, index) => (
        <div key={item.id} className={`flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold ${index <= position ? "bg-blue-600 text-white" : "bg-slate-200 text-slate-600"}`}>
          <span>{index + 1}</span>
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
};

const getConnectionTags = (fromNode, toId) => {
  if (!fromNode || !Array.isArray(fromNode.connections)) return [];
  const link = fromNode.connections.find((conn) => conn.to === toId);
  if (!link) return [];
  const tags = Array.isArray(link.tags) ? link.tags : [];
  return tags.filter(Boolean);
};

function Home() {
  const [stage, setStage] = useState(STAGES.UPLOAD);
  const [resumePreview, setResumePreview] = useState("");
  const [resumeFileName, setResumeFileName] = useState("");
  const [isExtracting, setIsExtracting] = useState(false);
  const [skills, setSkills] = useState(["Python", "React", "APIs", "Cloud"]);
  const [keywords, setKeywords] = useState(["alumni", "mentorship", "referrals"]);
  const [roles, setRoles] = useState(["Software Engineer", "DevOps Engineer"]);
  const [sourceName, setSourceName] = useState("Cris Huynh");
  const [targetName, setTargetName] = useState("Khoi Vu");
  const [targetCompany, setTargetCompany] = useState("RBC");
  const [jobTitle, setJobTitle] = useState("New-grad Software Developer");
  const [jobDescription, setJobDescription] = useState("");
  const [paths, setPaths] = useState([]);
  const [activePathIndex, setActivePathIndex] = useState(0);
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  const [stepStatuses, setStepStatuses] = useState([]);
  const [statusToast, setStatusToast] = useState("");
  const [error, setError] = useState("");
  const [info, setInfo] = useState("");
  const [signalsOpen, setSignalsOpen] = useState(false);

  useEffect(() => {
    if (!statusToast) return undefined;
    const timeout = setTimeout(() => setStatusToast(""), 2000);
    return () => clearTimeout(timeout);
  }, [statusToast]);

  useEffect(() => {
    return () => {
      if (resumePreview && resumePreview.startsWith("blob:")) {
        URL.revokeObjectURL(resumePreview);
      }
    };
  }, [resumePreview]);

  useEffect(() => {
    console.log("Paths updated:", paths);
    if (!paths.length) return;
    const current = paths[activePathIndex] || paths[0];
    const stepsCount = Math.max((current?.nodes?.length || 1) - 1, 0);
    setStepStatuses(Array.from({ length: stepsCount }, () => STATUS_STATES[0]));
    setActiveStepIndex(0);
    setSignalsOpen(false);
  }, [paths, activePathIndex]);

  const currentPath = paths[activePathIndex];
  const hopInfo = useMemo(() => {
    if (!currentPath) return [];
    const map = new Map();
    (currentPath.nodes_detail || []).forEach((node) => {
      if (node?._id) map.set(node._id, node);
    });
    return (currentPath.hops_detail || []).map((hop, index) => {
      const fromNode = map.get(hop.from) || {};
      const toNode = map.get(hop.to) || {};
      const connectionTags = getConnectionTags(fromNode, hop.to);
      return {
        index,
        from: fromNode.name || `Node ${index}`,
        to: toNode.name || `Node ${index + 1}`,
        signals: hop.signals || {},
        connectionTags,
      };
    });
  }, [currentPath]);

  const activeHop = hopInfo[activeStepIndex];
  const outreachSequence = useMemo(() => currentPath?.outreach_sequence || [], [currentPath]);
  const activeMessage = useMemo(() => {
    if (!activeHop) return "";
    const entry = outreachSequence.find((hop) => hop.step === activeHop.index + 1);
    return (entry?.message || "").trim();
  }, [outreachSequence, activeHop]);

  const fallbackMessage = useMemo(() => {
    if (!activeHop) return "";
    return (
      `Hi ${activeHop.to},\n\n` +
      `I noticed we both care about ${keywords.slice(0, 2).join(" & ")} and wanted to learn more about your work at ${targetCompany}. ` +
      `I'm pursuing ${jobTitle} roles and would appreciate a quick chat.`
    );
  }, [activeHop, keywords, targetCompany, jobTitle]);

  const handleResumeUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    setInfo("");
    setIsExtracting(true);
    setResumeFileName(file.name);

    const preview = URL.createObjectURL(file);
    setResumePreview(preview);

    try {
      const text = await file.text();
      const profile = await requestProfileExtraction(text);
      applyProfile(profile);
      setStage(STAGES.DETAILS);
    } catch (err) {
      setError(err.message || "Unable to read resume. Please try again with text/PDF.");
      setStage(STAGES.DETAILS);
    } finally {
      setIsExtracting(false);
    }
  };

  const requestProfileExtraction = async (text) => {
    const res = await fetch(`${API_BASE}/extract-profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Extraction failed");
    return data;
  };

  const applyProfile = (profile) => {
    if (!profile) return;
    if (profile.name) setSourceName(profile.name);
    if (Array.isArray(profile.skills) && profile.skills.length) setSkills(profile.skills);
    if (Array.isArray(profile.keywords) && profile.keywords.length) setKeywords(profile.keywords);
    if (profile.role) {
      const options = Array.from(new Set([profile.role, ...roles])).slice(0, 4);
      setRoles(options);
    }
  };

  const handleGenerateRoutes = async () => {
    setError("");
    setInfo("");
    setStage(STAGES.PATH_LOADING);
    try {
      const res = await fetch(`${API_BASE}/generate-routes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_name: sourceName, target_name: targetName }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Unable to generate routes");
      const fetched = Array.isArray(data.paths) ? data.paths.slice(0, 5) : [];
      setPaths(fetched);
      setActivePathIndex(0);
      setStage(STAGES.PATH_VIEW);
      if (!fetched.length) setInfo("No ≤6 hop paths found. Try adjusting skills or keywords.");
    } catch (err) {
      setError(err.message || "Failed to generate paths");
      setStage(STAGES.DETAILS);
    }
  };

  const handleStatusChange = (index, value) => {
    setStepStatuses((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  };

  const handleCopyMessage = async () => {
    const text = activeMessage || fallbackMessage;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setStatusToast("Message copied");
    } catch {
      setError("Clipboard unavailable in this browser.");
    }
  };

  const handleSaveStatus = () => setStatusToast("Status saved");

  const signalBadge = (signals = {}) => {
    if (signals.same_school) return "Same school";
    if (signals.same_company) return "Same company";
    if (typeof signals.skill_similarity === "number") return `Skill sim ${signals.skill_similarity.toFixed(2)}`;
    if (typeof signals.relationship_strength === "number" && signals.relationship_strength > 0) return "Warm mutual";
    return null;
  };

  const renderUploadStage = () => (
    <section className="flex h-full min-h-0 w-full items-start gap-10 rounded-[32px] bg-white p-10 shadow-xl">
      <div className="flex-1 space-y-10">
        <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">Step 1</p>
        <h2 className="text-3xl font-semibold text-slate-900">Input your resume here</h2>
        <label className="flex min-h-[24rem] w-full cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed border-slate-300 bg-slate-50/70 text-lg font-semibold text-slate-700 transition hover:border-indigo-400 hover:bg-indigo-50/60 md:min-h-[28rem]">
          <input type="file" accept=".txt,.pdf,.doc,.docx" className="hidden" onChange={handleResumeUpload} />
          <div className="flex flex-col items-center gap-3 px-6 text-center">
            <div className="rounded-2xl">
              <img src="/resume.png" alt="Resume upload icon" className="h-60 w-full object-contain" />
            </div>
            <div>
              <p className="text-base font-semibold text-slate-900">Drop your resume</p>
              <p className="text-xs font-normal text-slate-500">PDF, DOCX or TXT · Max 10 MB</p>
            </div>
            <span className="max-w-[240px] truncate rounded-full bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow">
              {resumeFileName ? `Selected: ${resumeFileName}` : "Browse files"}
            </span>
          </div>
        </label>
        <p className="text-sm text-slate-500">We only read the file on your device until you click upload.</p>
      </div>
      <div className="flex flex-1 h-full flex-col items-center justify-center gap-4 rounded-3xl bg-slate-50 p-6 text-lg font-semibold text-slate-700">
        <p>Extract your skills</p>
        {isExtracting ? (
          <AnimatedLoader title="Reading your resume" subtitle="Extracting skills, keywords and signals" icon={<img src="/resume.png" alt="Resume loader" className="h-60 w-60 p-3 object-contain" />} />
        ) : (
          <div className="flex flex-col items-center gap-2 text-center text-sm text-slate-500">
            <div className="relative h-16 w-16 mb-10">
              <div className="absolute inset-0 rounded-full bg-gradient-to-br from-slate-200 via-white to-slate-100 animate-pulse" />
              <div className="absolute inset-2 rounded-full bg-white shadow-inner" />
            </div>
            <p>Upload a resume to trigger extraction</p>
          </div>
        )}
      </div>
    </section>
  );

  const inputClasses = "w-full rounded-2xl border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200";

  const renderDetailsStage = () => (
    <section className="flex h-full min-h-0 w-full gap-8 rounded-[32px] bg-white p-8 shadow-xl">
      <div className="flex h-full w-[40%] min-w-[360px] items-center justify-center rounded-3xl bg-slate-100 p-6">
        {resumePreview ? (
          <iframe title="Resume preview" src={resumePreview} className="h-full w-full rounded-2xl border border-slate-200" />
        ) : (
          <img src="/resume.png" alt="Sample resume" className="max-h-full w-full rounded-2xl object-contain shadow-lg" />
        )}
      </div>
      <div className="flex flex-1 flex-col gap-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="text-sm text-slate-500">
            <span>From</span>
            <input className={inputClasses} value={sourceName} onChange={(event) => setSourceName(event.target.value)} />
          </label>
          <label className="text-sm text-slate-500">
            <span>Target contact</span>
            <input className={inputClasses} value={targetName} onChange={(event) => setTargetName(event.target.value)} />
          </label>
          <label className="text-sm text-slate-500">
            <span>Target company</span>
            <input className={inputClasses} value={targetCompany} onChange={(event) => setTargetCompany(event.target.value)} />
          </label>
          <label className="text-sm text-slate-500">
            <span>Target job title</span>
            <input className={inputClasses} value={jobTitle} onChange={(event) => setJobTitle(event.target.value)} />
          </label>
        </div>
        <TagEditor label="Skills" values={skills} onChange={setSkills} suggestions={SKILL_LIBRARY} />
        <TagEditor label="Keywords" values={keywords} onChange={setKeywords} suggestions={KEYWORD_LIBRARY} />
        <TagEditor label="What resume showed what you are?" values={roles} onChange={setRoles} suggestions={ROLE_LIBRARY} />
        <label className="space-y-2 text-sm text-slate-500">
          <span>Input Job Title and Description here</span>
          <textarea className={`${inputClasses} min-h-[160px]`} value={jobDescription} placeholder="Only type in this box. You can scroll inside." onChange={(event) => setJobDescription(event.target.value)} />
        </label>
        <div className="pt-2">
          <button type="button" className="w-full rounded-2xl bg-lime-400 px-4 py-3 text-base font-semibold text-lime-900 shadow hover:bg-lime-300" onClick={handleGenerateRoutes}>
            Generate connections
          </button>
        </div>
      </div>
    </section>
  );

  const renderPathLoadingStage = () => (
    <section className="flex h-full min-h-0 flex-col items-center justify-center gap-6 rounded-[32px] bg-white p-10 text-center shadow-xl">
      <AnimatedLoader title="Building your warm path" subtitle={`Scanning ${targetCompany}'s network for intros`} icon={<img src="/path.png" alt="Path loader" className="h-60 w-60 p-3 object-contain" />} />
      <div className="max-w-xl space-y-3 text-slate-700">
        <p className="text-base font-semibold text-slate-900">Extracting the shortest friendly route to {targetCompany}</p>
        <div className="grid gap-2 text-left text-sm text-slate-600">
          {["Checking shared schools and companies", "Ranking strongest mutual contacts", "Drafting outreach sequence"].map((item) => (
            <div key={item} className="flex items-center gap-3 rounded-2xl bg-slate-100 px-3 py-2">
              <span className="relative flex h-3 w-3">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-lime-400 opacity-75" />
                <span className="relative inline-flex h-3 w-3 rounded-full bg-lime-500" />
              </span>
              <span>{item}</span>
            </div>
          ))}
        </div>
        <p className="text-xs italic text-slate-500">Hang tight - this takes just a few seconds.</p>
        <div className="h-auto w-full bg-slate-200 p-3 rounded-lg m-4" >
          <p className="text-md italic text-red-500 font-semibold">Fact: Employers are hiring at one of the slowest paces in over a decade, even as the official unemployment rate remains relatively low</p>
        </div>
      </div>
    </section>
  );

  const renderPathStage = () => (
    <section className="grid h-full min-h-0 w-full gap-8 rounded-[32px] bg-white p-8 shadow-xl lg:grid-cols-[1.1fr,0.9fr]">
      <div className="flex h-full min-h-0 flex-col">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">Path</p>
            <h2 className="text-2xl font-semibold text-slate-900">
              {sourceName} → {targetName}
            </h2>
          </div>
          {!!paths.length && (
            <div className="flex flex-wrap gap-2">
              {paths.map((path, index) => (
                <button
                  type="button"
                  key={`path-${index}`}
                  className={`rounded-full border px-3 py-1 text-xs font-semibold ${index === activePathIndex ? "border-slate-900 bg-slate-900 text-white" : "border-slate-200 bg-white text-slate-700"}`}
                  onClick={() => setActivePathIndex(index)}
                >
                  Path {index + 1} · score {path.score?.toFixed(2)}
                </button>
              ))}
            </div>
          )}
        </div>
        {paths.length === 0 ? (
          <div className="mt-6 rounded-2xl border border-dashed border-slate-200 p-8 text-center text-slate-500">
            <p>No routes within 6 hops were found.</p>
            <p className="text-sm text-slate-400">Try adding different keywords or another school/company signal.</p>
          </div>
        ) : (
          <div className="mt-2 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
            {hopInfo.map((hop, index) => {
              return (
                <button
                  key={`${hop.from}-${hop.to}-${index}`}
                  type="button"
                  className={`rounded-2xl border px-4 py-3 text-left transition ${index === activeStepIndex ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-slate-50 hover:bg-white"}`}
                  onClick={() => {
                    setActiveStepIndex(index);
                    setSignalsOpen(false);
                  }}
                >
                  <p className="text-xs uppercase tracking-widest text-slate-500">Step {index + 1}</p>
                  <p className="text-lg font-semibold text-slate-900">
                    {hop.from} → {hop.to}
                  </p>
                  {signalBadge(hop.signals) && <span className="mb-2 inline-block rounded-full bg-gray-700 px-3 py-1 text-xs text-white font-bold">{signalBadge(hop.signals)}</span>}
                  <p className="text-xs m-1 text-slate-500">{stepStatuses[index] || STATUS_STATES[0]}</p>
                </button>
              );
            })}
          </div>
        )}
      </div>
      <div className="flex h-full min-h-0 flex-col rounded-3xl border border-slate-200 bg-slate-50 p-6">
        {activeHop ? (
          <div className="flex min-h-0 flex-1 flex-col space-y-4 overflow-y-auto">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-lg font-semibold text-slate-900">Outreach message – Step {activeHop.index + 1}</p>
              <button type="button" className="text-sm text-blue-600 underline-offset-4 hover:underline" onClick={() => setSignalsOpen((prev) => !prev)}>
                {signalsOpen ? "Hide shared signals" : "Expand shared signals"}
              </button>
            </div>
            <p className="text-sm text-slate-500">
              From: {sourceName} → {activeHop.to}
            </p>
            {signalsOpen && (
              <ul className="list-disc space-y-1 rounded-2xl bg-white p-4 text-sm text-slate-700">
                {buildSignalList(activeHop.signals, activeHop.connectionTags).map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            )}
            <label className="space-y-1 text-sm text-slate-500">
              <span>Step status</span>
              <select className="w-full rounded-2xl border border-slate-200 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200" value={stepStatuses[activeHop.index] || STATUS_STATES[0]} onChange={(event) => handleStatusChange(activeHop.index, event.target.value)}>
                {STATUS_STATES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </label>
            <div className="space-y-2 rounded-2xl bg-white p-4">
              <div className="flex items-center justify-between text-sm">
                <p className="font-semibold text-slate-900">Suggested outreach message</p>
                <button type="button" className="text-blue-600 underline-offset-4 hover:underline" onClick={handleCopyMessage}>
                  Copy message
                </button>
              </div>
              <textarea className="h-48 w-full rounded-2xl border border-slate-200 p-3 text-sm text-slate-900" value={activeMessage || fallbackMessage} readOnly />
            </div>
            <div className="flex flex-wrap gap-3 pt-2">
              <button type="button" className="rounded-2xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white" onClick={handleSaveStatus}>
                Save status
              </button>
              <button type="button" className="text-sm text-slate-600 underline-offset-4 hover:underline" onClick={() => setStage(STAGES.DETAILS)}>
                Back to edit
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-slate-200 p-8 text-center text-slate-500">
            <p>Select a step on the left to view outreach guidance.</p>
          </div>
        )}
      </div>
    </section>
  );

  const renderStage = () => {
    switch (stage) {
      case STAGES.UPLOAD:
        return renderUploadStage();
      case STAGES.DETAILS:
        return renderDetailsStage();
      case STAGES.PATH_LOADING:
        return renderPathLoadingStage();
      case STAGES.PATH_VIEW:
        return renderPathStage();
      default:
        return null;
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-slate-50 text-slate-900">
      <div className="mx-auto flex h-full w-full max-w-[1700px] flex-col px-8 py-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">6-Step Connection Builder</p>
            <h1 className="text-3xl font-semibold text-slate-900">Welcome {sourceName || "there"}</h1>
          </div>
          <StageHeading stage={stage} />
        </div>
        {error && (
          <div className="flex items-center justify-between rounded-2xl bg-rose-100 px-4 py-1 text-sm font-medium text-rose-800">
            <p>{error}</p>
            <button type="button" onClick={() => setError("")}>
              ×
            </button>
          </div>
        )}
        {info && (
          <div className="flex items-center justify-between rounded-2xl bg-blue-100 px-4 py-3 text-sm font-medium text-blue-800">
            <p>{info}</p>
            <button type="button" onClick={() => setInfo("")}>
              ×
            </button>
          </div>
        )}
        {statusToast && <div className="rounded-2xl bg-emerald-100 px-4 py-3 text-sm font-medium text-emerald-800">{statusToast}</div>}
        <div className="min-h-0 flex-1">{renderStage()}</div>
      </div>
    </div>
  );
}

export default Home;
