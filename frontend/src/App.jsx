import { useCallback, useEffect, useState } from "react";
import {
  Alert,
  AppBar,
  Backdrop,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  Grid,
  LinearProgress,
  Stack,
  TextField,
  Toolbar,
  Typography,
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DownloadIcon from "@mui/icons-material/Download";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import InsightsIcon from "@mui/icons-material/Insights";
import TipsAndUpdatesIcon from "@mui/icons-material/TipsAndUpdates";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";
const COVER_IMAGE_URL = `${API_BASE_URL}/cover-image`;
const acceptedExtensions = ["pdf", "docx"];

const glassCardSx = {
  borderRadius: 4,
  background: "rgba(9, 26, 35, 0.78)",
  border: "1px solid rgba(99, 182, 171, 0.35)",
  boxShadow: "0 18px 48px rgba(1, 12, 18, 0.35)",
  backdropFilter: "blur(12px)",
};

const panelCardSx = {
  borderRadius: 4,
  background: "rgba(250, 253, 252, 0.97)",
  border: "1px solid rgba(18, 78, 78, 0.14)",
  boxShadow: "0 20px 38px rgba(7, 34, 39, 0.2)",
};

const scorePalette = {
  high: "#21b97b",
  medium: "#f0b035",
  low: "#e85353",
};

const getScoreColor = (value) => {
  if (value >= 75) return scorePalette.high;
  if (value >= 50) return scorePalette.medium;
  return scorePalette.low;
};

const formatPercent = (value) => Number(value || 0).toFixed(1);
const formatSectionLabel = (label = "") =>
  label
    .split(" ")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : ""))
    .join(" ");

const MetricBar = ({ label, value }) => {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0));
  const color = getScoreColor(safeValue);
  return (
    <Box sx={{ mb: 1.2, width: "100%" }}>
      <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.3 }}>
        <Typography sx={{ fontWeight: 600, color: "#365158" }}>{label}</Typography>
        <Typography sx={{ fontWeight: 700, color }}>{safeValue.toFixed(1)}%</Typography>
      </Stack>
      <LinearProgress
        variant="determinate"
        value={safeValue}
        sx={{
          height: 8,
          borderRadius: 999,
          bgcolor: "rgba(12, 52, 58, 0.12)",
          "& .MuiLinearProgress-bar": { bgcolor: color },
        }}
      />
    </Box>
  );
};

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadedMeta, setUploadedMeta] = useState([]);
  const [jobDescription, setJobDescription] = useState("");
  const [results, setResults] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [jobSkills, setJobSkills] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [advisorResumeText, setAdvisorResumeText] = useState("");
  const [advisorJobDescription, setAdvisorJobDescription] = useState("");
  const [advisorSuggestions, setAdvisorSuggestions] = useState([]);
  const [advisorMissingSkills, setAdvisorMissingSkills] = useState([]);
  const [advisorLoading, setAdvisorLoading] = useState(false);
  const [advisorError, setAdvisorError] = useState("");
  const [activeCandidateIndex, setActiveCandidateIndex] = useState(0);
  const [coverAvailable, setCoverAvailable] = useState(false);
  const [advisorMode, setAdvisorMode] = useState("auto");
  const [advisorCache, setAdvisorCache] = useState({});
  const [advisorLinkedCandidate, setAdvisorLinkedCandidate] = useState(null);

  useEffect(() => {
    setActiveCandidateIndex(0);
  }, [results.length]);

  useEffect(() => {
    let canceled = false;
    const controller = new AbortController();
    fetch(COVER_IMAGE_URL, { method: "HEAD", signal: controller.signal })
      .then((response) => {
        if (!canceled) {
          setCoverAvailable(response.ok);
        }
      })
      .catch(() => {
        if (!canceled) {
          setCoverAvailable(false);
        }
      });

    return () => {
      canceled = true;
      controller.abort();
    };
  }, []);

  const topCandidates = analytics?.top_candidates || [];
  const averageOverall = analytics?.average_overall_score ?? 0;
  const averageSemantic = analytics?.average_semantic_score ?? 0;
  const skillDistribution = (analytics?.skill_distribution || []).slice(0, 10);
  const activeCandidate = results[activeCandidateIndex] || null;
  const averageScoreLabel = formatPercent(averageOverall);
  const candidateFile = activeCandidate?.file_name || "";
  const candidateName = activeCandidate?.candidate_name || "";

  const validateFiles = (files) => {
    const valid = files.filter((file) => {
      const ext = file.name.split(".").pop()?.toLowerCase();
      return acceptedExtensions.includes(ext);
    });

    if (valid.length !== files.length) {
      setWarnings((prev) => [...prev, "Some files were ignored. Only PDF and DOCX are allowed."]);
    }

    return valid;
  };

  const onFileChange = (event) => {
    const files = validateFiles(Array.from(event.target.files || []));
    setSelectedFiles(files);
    setError("");
    setSuccessMessage("");
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const files = validateFiles(Array.from(event.dataTransfer.files || []));
    setSelectedFiles(files);
  };

  const onDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const uploadResumes = async () => {
    if (selectedFiles.length === 0) {
      throw new Error("Please select at least one valid PDF or DOCX resume.");
    }

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("resumes", file));

    const response = await fetch(`${API_BASE_URL}/upload-resumes`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Upload failed.");
    }

    setUploadedMeta(data.uploaded || []);
    return data;
  };

  const analyzeCandidates = async () => {
    if (!jobDescription.trim()) {
      setError("Please enter a job description before analysis.");
      return;
    }

    try {
      setIsAnalyzing(true);
      setError("");
      setWarnings([]);

      const uploadData = await uploadResumes();

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_description: jobDescription,
          uploaded_files: (uploadData.uploaded || []).map((item) => item.stored_name),
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Analysis failed.");
      }

      setResults(data.results || []);
      setAnalytics(data.analytics || null);
      setJobSkills(data.job_description_skills || []);
      setWarnings([...(uploadData.errors || []), ...(data.warnings || [])]);
      setSuccessMessage(`Analysis completed. ${data.total_candidates || 0} candidates ranked.`);
      setAdvisorMode("auto");
      setAdvisorCache({});
      setAdvisorLinkedCandidate(null);
      setAdvisorSuggestions([]);
      setAdvisorMissingSkills([]);
      setAdvisorError("");
    } catch (analysisError) {
      setResults([]);
      setAnalytics(null);
      const friendlyError =
        analysisError?.message === "Failed to fetch"
          ? `Unable to reach the API at ${API_BASE_URL}. Make sure the Flask backend is running and accessible.`
          : analysisError?.message || "Unable to analyze candidates.";
      setError(friendlyError);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadCsv = () => {
    window.open(`${API_BASE_URL}/download-results`, "_blank");
  };

  const relinkAdvisorToCandidate = () => {
    setAdvisorMode("auto");
    setAdvisorResumeText("");
    setAdvisorJobDescription("");
    setAdvisorLinkedCandidate(candidateName || null);
    setAdvisorError("");
  };

  const requestAdvisorSuggestions = useCallback(
    async (options = {}) => {
      const {
        resumeText: overrideResume,
        jdText: overrideJobDescription,
        candidateFile,
        candidateName,
        cacheKey,
      } = options;

      const resolvedJobDescriptionSource =
        overrideJobDescription ?? advisorJobDescription ?? jobDescription ?? "";
      const resolvedResumeTextSource = overrideResume ?? advisorResumeText ?? "";
      const resolvedJobDescription = resolvedJobDescriptionSource.trim();
      const resolvedResumeText = resolvedResumeTextSource.trim();
      const usingCandidateLink = Boolean(candidateFile);

      if (!resolvedJobDescription) {
        setAdvisorError("Provide a job description before generating suggestions.");
        return;
      }

      if (!usingCandidateLink && !resolvedResumeText) {
        setAdvisorError("Provide resume text or link to a candidate to generate suggestions.");
        return;
      }

      setAdvisorError("");
      setAdvisorSuggestions([]);
      setAdvisorMissingSkills([]);

      try {
        setAdvisorLoading(true);
        const payload = { job_description: resolvedJobDescription };
        if (usingCandidateLink) {
          payload.candidate_file = candidateFile;
        } else {
          payload.resume_text = resolvedResumeText;
        }

        const response = await fetch(`${API_BASE_URL}/suggest-improvements`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Unable to generate suggestions.");
        }

        setAdvisorSuggestions(data.suggestions || []);
        setAdvisorMissingSkills(data.missing_skills || []);

        if (usingCandidateLink) {
          setAdvisorLinkedCandidate(candidateName || candidateFile);
          setAdvisorCache((prev) => ({
            ...prev,
            [(cacheKey || candidateFile)]: data,
          }));
        } else {
          setAdvisorLinkedCandidate(null);
        }
      } catch (advisorErr) {
        setAdvisorError(advisorErr.message || "Unable to generate suggestions.");
      } finally {
        setAdvisorLoading(false);
      }
    },
    [advisorResumeText, advisorJobDescription, jobDescription]
  );

  useEffect(() => {
    if (advisorMode !== "auto") {
      return;
    }

    const trimmedJobDescription = jobDescription.trim();
    if (!candidateFile || !trimmedJobDescription) {
      return;
    }

    const cacheKey = `${candidateFile}::${trimmedJobDescription}`;
    const cachedEntry = advisorCache[cacheKey];

    if (cachedEntry) {
      setAdvisorLinkedCandidate(candidateName || candidateFile);
      setAdvisorSuggestions(cachedEntry.suggestions || []);
      setAdvisorMissingSkills(cachedEntry.missing_skills || []);
      setAdvisorError("");
      return;
    }

    requestAdvisorSuggestions({
      candidateFile,
      candidateName,
      jdText: trimmedJobDescription,
      cacheKey,
    });
  }, [advisorMode, candidateFile, candidateName, jobDescription, advisorCache, requestAdvisorSuggestions]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        pb: { xs: 4, md: 7 },
        background: "linear-gradient(145deg, #04181f 0%, #0a3d49 30%, #61a5ab 85%)",
      }}
    >
      <Box
        sx={{
          position: "fixed",
          inset: 0,
          pointerEvents: "none",
          opacity: coverAvailable ? 0.18 : 0,
          backgroundImage: coverAvailable ? `url(${COVER_IMAGE_URL})` : "none",
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
          backgroundSize: "cover",
          filter: "grayscale(0.1)",
        }}
      />

      <Box
        sx={{
          position: "fixed",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(60% 40% at 20% 5%, rgba(255, 229, 152, 0.35), transparent 80%), radial-gradient(35% 50% at 80% 70%, rgba(11, 130, 121, 0.22), transparent 75%)",
        }}
      />

      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          background: "rgba(4, 24, 31, 0.8)",
          backdropFilter: "blur(10px)",
          borderBottom: "1px solid rgba(146, 225, 201, 0.28)",
        }}
      >
        <Toolbar sx={{ py: 1.4 }}>
          <AutoAwesomeIcon sx={{ color: "#fef3bf", mr: 1.2 }} />
          <Box>
            <Typography
              variant="h5"
              sx={{
                color: "#f3fffc",
                fontFamily: "'Sora', 'Manrope', sans-serif",
                fontWeight: 700,
                letterSpacing: "0.015em",
              }}
            >
              Smart Resume Screener
            </Typography>
            <Typography sx={{ color: "rgba(233, 251, 248, 0.8)", fontSize: 13 }}>
              AI-assisted candidate analysis & insights
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: { xs: 3, md: 4 }, position: "relative", zIndex: 1 }}>
        <Grid container spacing={3.2}>
          <Grid item xs={12} lg={5} className="float-in">
            <Stack spacing={3}>
              <Card sx={glassCardSx}>
                <CardContent sx={{ p: { xs: 2.2, sm: 3 } }}>
                  <Typography
                    variant="h5"
                    sx={{ color: "#dffaf3", fontFamily: "'Sora', sans-serif", fontWeight: 700, mb: 2.4 }}
                  >
                    Candidate Screening
                  </Typography>

                  <Box
                    onDrop={onDrop}
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    sx={{
                      border: "2px dashed",
                      borderColor: isDragging ? "#9ee8d8" : "rgba(188, 246, 227, 0.45)",
                      borderRadius: 3,
                      p: { xs: 3, sm: 4 },
                      textAlign: "center",
                      bgcolor: isDragging ? "rgba(225, 255, 244, 0.18)" : "rgba(9, 53, 63, 0.42)",
                      transition: "all 240ms ease",
                      "&:hover": {
                        borderColor: "#9ee8d8",
                        bgcolor: "rgba(225, 255, 244, 0.14)",
                      },
                    }}
                  >
                    <CloudUploadIcon sx={{ fontSize: 48, color: "#d8fff2", mb: 1.3 }} />
                    <Typography sx={{ fontWeight: 700, color: "#f4fffb", fontSize: 27, lineHeight: 1.15 }}>
                      Drag resumes here
                    </Typography>
                    <Typography sx={{ color: "rgba(224, 248, 240, 0.86)", mt: 0.6, mb: 2.2 }}>
                      Accepts PDF and DOCX files
                    </Typography>
                    <Button
                      variant="contained"
                      component="label"
                      sx={{
                        textTransform: "none",
                        borderRadius: 99,
                        fontWeight: 700,
                        px: 3.2,
                        py: 1,
                        bgcolor: "#f6d786",
                        color: "#09353f",
                        boxShadow: "0 8px 18px rgba(253, 220, 126, 0.34)",
                        "&:hover": { bgcolor: "#f4c85d" },
                      }}
                    >
                      Browse Files
                      <input hidden type="file" multiple accept=".pdf,.docx" onChange={onFileChange} />
                    </Button>
                  </Box>

                  {selectedFiles.length > 0 && (
                    <Box sx={{ mt: 2.2 }}>
                      <Typography sx={{ color: "#dbfef6", fontWeight: 600, mb: 1.1 }}>
                        Uploaded Files ({selectedFiles.length})
                      </Typography>
                      <Stack spacing={1}>
                        {selectedFiles.map((file, index) => (
                          <Stack
                            key={`${file.name}-${index}`}
                            direction="row"
                            justifyContent="space-between"
                            alignItems="center"
                            sx={{
                              p: 1.1,
                              borderRadius: 2,
                              bgcolor: "rgba(220, 255, 244, 0.17)",
                              border: "1px solid rgba(183, 239, 221, 0.45)",
                            }}
                          >
                            <Typography sx={{ color: "#e8fff8", fontWeight: 600 }}>
                              {file.name}
                            </Typography>
                            <Chip
                              size="small"
                              label={`${(file.size / 1024).toFixed(1)} KB`}
                              sx={{ bgcolor: "rgba(248, 215, 126, 0.25)", color: "#fceab5", fontWeight: 700 }}
                            />
                          </Stack>
                        ))}
                      </Stack>
                    </Box>
                  )}

                  <TextField
                    label="Job Description"
                    multiline
                    minRows={7}
                    fullWidth
                    value={jobDescription}
                    onChange={(event) => setJobDescription(event.target.value)}
                    sx={{
                      mt: 2.2,
                      "& .MuiInputLabel-root": { color: "rgba(228, 249, 241, 0.86)" },
                      "& .MuiOutlinedInput-root": {
                        color: "#f2fffb",
                        borderRadius: 2.4,
                        background: "rgba(7, 49, 59, 0.45)",
                        "& fieldset": { borderColor: "rgba(179, 235, 219, 0.45)" },
                        "&:hover fieldset": { borderColor: "rgba(205, 248, 234, 0.7)" },
                        "&.Mui-focused fieldset": { borderColor: "#9ee8d8" },
                      },
                    }}
                  />

                  <Stack direction="row" spacing={1.2} sx={{ mt: 2.2 }}>
                    <Button
                      fullWidth
                      variant="contained"
                      onClick={analyzeCandidates}
                      disabled={isAnalyzing}
                      sx={{
                        textTransform: "none",
                        borderRadius: 2,
                        fontWeight: 700,
                        bgcolor: "#f6d786",
                        color: "#08343c",
                        "&:hover": { bgcolor: "#f4c85d" },
                      }}
                    >
                      Analyze Candidates
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<DownloadIcon />}
                      onClick={downloadCsv}
                      disabled={results.length === 0}
                      sx={{
                        minWidth: 92,
                        textTransform: "none",
                        borderRadius: 2,
                        color: "#eafdf7",
                        borderColor: "rgba(215, 253, 239, 0.4)",
                        "&:hover": { borderColor: "#b8f8e6", bgcolor: "rgba(236, 238, 237, 0.08)" },
                      }}
                    >
                      CSV
                    </Button>
                  </Stack>

                  {isAnalyzing && (
                    <LinearProgress sx={{ mt: 2.1, borderRadius: 999, height: 6, bgcolor: "rgba(255,255,255,0.2)" }} />
                  )}

                  {!!error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
                  {!!successMessage && <Alert severity="success" sx={{ mt: 2 }}>{successMessage}</Alert>}
                  {warnings.length > 0 && (
                    <Stack spacing={1} sx={{ mt: 2 }}>
                      {warnings.map((warning, index) => (
                        <Alert key={`${warning}-${index}`} severity="warning">
                          {warning}
                        </Alert>
                      ))}
                    </Stack>
                  )}
                </CardContent>
              </Card>

              <Card sx={{ ...glassCardSx, background: "rgba(8, 28, 36, 0.8)" }}>
                <CardContent sx={{ p: { xs: 2.2, sm: 2.8 } }}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.2 }}>
                    <TipsAndUpdatesIcon sx={{ color: "#f6d786" }} />
                    <Typography sx={{ color: "#f1fffb", fontWeight: 700, fontSize: 18 }}>
                      Resume Improvement Advisor
                    </Typography>
                  </Stack>
                  <Stack
                    direction={{ xs: "column", sm: "row" }}
                    spacing={1}
                    justifyContent="space-between"
                    alignItems={{ xs: "flex-start", sm: "center" }}
                    sx={{ mb: 1.6 }}
                  >
                    <Typography sx={{ color: "rgba(231, 252, 247, 0.75)" }}>
                      Automatically surfaces suggestions for the selected candidate or paste custom text to override.
                    </Typography>
                    {advisorMode === "manual" ? (
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={relinkAdvisorToCandidate}
                        sx={{
                          textTransform: "none",
                          borderRadius: 999,
                          borderColor: "rgba(167, 229, 215, 0.6)",
                          color: "#bff4ea",
                          px: 2,
                        }}
                      >
                        Link to Selected Candidate
                      </Button>
                    ) : (
                      advisorLinkedCandidate && (
                        <Chip
                          label={`Auto: ${advisorLinkedCandidate}`}
                          size="small"
                          sx={{ bgcolor: "rgba(158, 232, 216, 0.2)", color: "#9ee8d8", fontWeight: 700 }}
                        />
                      )
                    )}
                  </Stack>

                  <TextField
                    label="Resume Text"
                    multiline
                    minRows={4}
                    fullWidth
                    value={advisorResumeText}
                    placeholder={
                      advisorMode === "auto"
                        ? advisorLinkedCandidate
                          ? `Auto-linked to ${advisorLinkedCandidate}'s resume`
                          : "Auto mode active. Select a candidate or type to override."
                        : "Paste resume text to manually request suggestions."
                    }
                    onChange={(event) => {
                      setAdvisorMode("manual");
                      setAdvisorLinkedCandidate(null);
                      setAdvisorResumeText(event.target.value);
                    }}
                    sx={{
                      mb: 1.4,
                      "& .MuiInputBase-root": { background: "rgba(3, 22, 30, 0.65)", color: "#f4fffb" },
                      "& fieldset": { borderColor: "rgba(167, 229, 215, 0.35)" },
                    }}
                  />

                  <TextField
                    label="Advisor Job Description (optional)"
                    multiline
                    minRows={2}
                    fullWidth
                    value={advisorJobDescription}
                    placeholder="Defaults to the main job description above"
                    onChange={(event) => {
                      setAdvisorMode("manual");
                      setAdvisorLinkedCandidate(null);
                      setAdvisorJobDescription(event.target.value);
                    }}
                    sx={{
                      mb: 1.6,
                      "& .MuiInputBase-root": { background: "rgba(3, 22, 30, 0.65)", color: "#f4fffb" },
                      "& fieldset": { borderColor: "rgba(167, 229, 215, 0.35)" },
                    }}
                  />

                  <Button
                    variant="contained"
                    onClick={requestAdvisorSuggestions}
                    disabled={advisorLoading}
                    sx={{
                      textTransform: "none",
                      borderRadius: 999,
                      fontWeight: 700,
                      bgcolor: "#9ee8d8",
                      color: "#0a2a2f",
                      "&:hover": { bgcolor: "#7fd7c9" },
                    }}
                  >
                    Generate Suggestions
                  </Button>

                  {advisorLoading && <LinearProgress sx={{ mt: 1.5, borderRadius: 999 }} />}
                  {!!advisorError && <Alert severity="error" sx={{ mt: 2 }}>{advisorError}</Alert>}

                  {advisorMissingSkills.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography sx={{ color: "#fce9c4", fontWeight: 700, mb: 0.8 }}>
                        Missing Skills to Highlight
                      </Typography>
                      <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap">
                        {advisorMissingSkills.map((skill) => (
                          <Chip key={skill} label={skill} size="small" sx={{ bgcolor: "rgba(255, 255, 255, 0.08)", color: "#ffe2a7" }} />
                        ))}
                      </Stack>
                    </Box>
                  )}

                  {advisorSuggestions.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography sx={{ color: "#bff4ea", fontWeight: 700, mb: 0.8 }}>Actionable Suggestions</Typography>
                      <Stack spacing={0.8}>
                        {advisorSuggestions.map((tip, index) => (
                          <Typography key={`${tip}-${index}`} sx={{ color: "rgba(234, 254, 249, 0.92)" }}>
                            • {tip}
                          </Typography>
                        ))}
                      </Stack>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          <Grid item xs={12} lg={7}>
            <Stack spacing={2.4}>
              <Card sx={panelCardSx} className="float-in delayed">
                <CardContent>
                  <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={2}>
                    <Box>
                      <Typography
                        variant="h5"
                        sx={{ color: "#0b4a53", fontWeight: 700, fontFamily: "'Sora', sans-serif", mb: 1.2 }}
                      >
                        Dashboard Insights
                      </Typography>
                      <Typography sx={{ color: "#4d6a6f" }}>
                        Track how closely your applicant pool aligns with the job description.
                      </Typography>
                    </Box>
                    <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
                      <Chip label={`Uploads: ${uploadedMeta.length}`} sx={{ bgcolor: "#e2f8f1", color: "#1a695e", fontWeight: 700 }} />
                      <Chip label={`Top Candidates: ${topCandidates.length}`} sx={{ bgcolor: "#e2edff", color: "#1e4c92", fontWeight: 700 }} />
                      <Chip label={`Avg Score: ${averageScoreLabel}%`} sx={{ bgcolor: "#fff3cb", color: "#78590d", fontWeight: 700 }} />
                    </Stack>
                  </Stack>

                  <Stack direction={{ xs: "column", sm: "row" }} spacing={1.4} sx={{ mt: 2 }}>
                    <Card sx={{ flex: 1, background: "#f0fff8", borderRadius: 3, boxShadow: "none", p: 2 }}>
                      <Typography sx={{ color: "#1b5c53", fontWeight: 600 }}>Average Overall Match</Typography>
                      <Typography sx={{ fontSize: 30, fontWeight: 800, color: getScoreColor(averageOverall) }}>
                        {formatPercent(averageOverall)}%
                      </Typography>
                    </Card>
                    <Card sx={{ flex: 1, background: "#fff7e1", borderRadius: 3, boxShadow: "none", p: 2 }}>
                      <Typography sx={{ color: "#a76b06", fontWeight: 600 }}>Average Semantic Score</Typography>
                      <Typography sx={{ fontSize: 30, fontWeight: 800, color: getScoreColor(averageSemantic) }}>
                        {formatPercent(averageSemantic)}%
                      </Typography>
                    </Card>
                  </Stack>

                  <Divider sx={{ my: 2 }} />
                  <Typography sx={{ fontWeight: 700, color: "#1f4247", mb: 1 }}>Required JD Skills</Typography>
                  <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap">
                    {jobSkills.length ? (
                      jobSkills.map((skill) => (
                        <Chip key={skill} size="small" label={skill} sx={{ bgcolor: "#dff6ee", color: "#17685f", fontWeight: 700 }} />
                      ))
                    ) : (
                      <Chip size="small" label="No JD skills detected yet" sx={{ bgcolor: "#ecf5f2", color: "#587176" }} />
                    )}
                  </Stack>
                </CardContent>
              </Card>

              <Card sx={panelCardSx}>
                <CardContent>
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.2 }}>
                    <InsightsIcon sx={{ color: "#0a7b83" }} />
                    <Typography sx={{ color: "#0f3a45", fontWeight: 700, fontSize: 18 }}>Analytics Panel</Typography>
                  </Stack>
                  <Typography sx={{ color: "#4d6a6f", mb: 1.4 }}>
                    Understand who is rising to the top and which skills dominate the applicant pool.
                  </Typography>

                  <Typography sx={{ fontWeight: 700, color: "#1f4247", mb: 0.6 }}>Top Candidates</Typography>
                  {topCandidates.length ? (
                    <Stack spacing={0.6}>
                      {topCandidates.slice(0, 3).map((candidate) => (
                        <Stack
                          key={candidate.file_name}
                          direction="row"
                          justifyContent="space-between"
                          sx={{ p: 1, borderRadius: 2, bgcolor: "#f4fafa" }}
                        >
                          <Typography sx={{ fontWeight: 600, color: "#173f45" }}>{candidate.candidate_name}</Typography>
                          <Chip
                            label={`${formatPercent(candidate.overall_match)}%`}
                            size="small"
                            sx={{ bgcolor: "rgba(33, 185, 123, 0.15)", color: "#17805a", fontWeight: 700 }}
                          />
                        </Stack>
                      ))}
                    </Stack>
                  ) : (
                    <Typography sx={{ color: "#5f7074" }}>Run an analysis to populate insights.</Typography>
                  )}

                  <Divider sx={{ my: 2 }} />
                  <Typography sx={{ fontWeight: 700, color: "#1f4247", mb: 0.6 }}>Skills Distribution</Typography>
                  <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap">
                    {skillDistribution.length ? (
                      skillDistribution.map((skill) => (
                        <Chip
                          key={skill.skill}
                          label={`${skill.skill} · ${skill.count}`}
                          size="small"
                          sx={{ bgcolor: "#e8f3ff", color: "#1f4f91", fontWeight: 600 }}
                        />
                      ))
                    ) : (
                      <Chip size="small" label="No data yet" sx={{ bgcolor: "#f1f1f1", color: "#5c6b6f" }} />
                    )}
                  </Stack>
                </CardContent>
              </Card>

              {activeCandidate && (
                <Card sx={panelCardSx}>
                  <CardContent>
                    <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                      <TipsAndUpdatesIcon sx={{ color: "#f0a500" }} />
                      <Typography sx={{ fontWeight: 700, color: "#143c45", fontSize: 18 }}>
                        Suggestions Panel · {activeCandidate.candidate_name}
                      </Typography>
                    </Stack>
                    <Typography sx={{ color: "#4d6a6f", mb: 1.3 }}>{activeCandidate.explanation}</Typography>
                    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mb: 1 }}>
                      <Chip label={`Overall: ${formatPercent(activeCandidate.overall_match)}%`} sx={{ bgcolor: "#e2f8f1", color: "#1a695e", fontWeight: 700 }} />
                      <Chip label={`Semantic: ${formatPercent(activeCandidate.semantic_similarity)}%`} sx={{ bgcolor: "#fff1db", color: "#a66b05", fontWeight: 700 }} />
                      <Chip label={`TF-IDF: ${formatPercent(activeCandidate.tfidf_score)}%`} sx={{ bgcolor: "#e6f1ff", color: "#1c4e8e", fontWeight: 700 }} />
                      {activeCandidate?.recommended_role?.role && (
                        <Chip
                          label={`Best suited: ${activeCandidate.recommended_role.role}`}
                          sx={{ bgcolor: "rgba(33, 185, 123, 0.18)", color: "#127055", fontWeight: 700 }}
                        />
                      )}
                    </Stack>

                    {activeCandidate?.section_scores && (
                      <Box sx={{ mb: 1.2 }}>
                        <Typography sx={{ fontWeight: 700, color: "#1c4c53", mb: 0.6 }}>Section Scores</Typography>
                        <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap">
                          {Object.entries(activeCandidate.section_scores).map(([section, data]) => (
                            <Chip
                              key={`active-${section}`}
                              label={`${formatSectionLabel(section)}: ${formatPercent(data?.score || 0)}%`}
                              sx={{ bgcolor: "rgba(11, 58, 63, 0.08)", color: "#1c5157", fontWeight: 600 }}
                            />
                          ))}
                        </Stack>
                      </Box>
                    )}
                    <Typography sx={{ fontWeight: 700, color: "#1c4c53", mb: 0.6 }}>Missing Skills</Typography>
                    <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap" sx={{ mb: 1.2 }}>
                      {(activeCandidate.missing_skills || []).length ? (
                        activeCandidate.missing_skills.map((skill) => (
                          <Chip key={skill} label={skill} size="small" sx={{ bgcolor: "#fde3df", color: "#9f3a31", fontWeight: 700 }} />
                        ))
                      ) : (
                        <Chip size="small" label="All required skills present" sx={{ bgcolor: "#e0f6ec", color: "#1f785f", fontWeight: 700 }} />
                      )}
                    </Stack>

                    {(activeCandidate.suggestions || []).length > 0 && (
                      <Stack spacing={0.6}>
                        {activeCandidate.suggestions.map((tip, index) => (
                          <Typography key={`${tip}-${index}`} sx={{ color: "#2a5054" }}>
                            • {tip}
                          </Typography>
                        ))}
                      </Stack>
                    )}
                  </CardContent>
                </Card>
              )}

              {results.length === 0 ? (
                <Card sx={panelCardSx} className="float-in delayed-more">
                  <CardContent sx={{ py: 3.5 }}>
                    <Typography sx={{ color: "#264e53", fontSize: 25, fontWeight: 600 }}>
                      No ranked results yet.
                    </Typography>
                    <Typography sx={{ color: "#4e7074", mt: 0.4 }}>
                      Upload resumes and run analysis to see ranked candidates.
                    </Typography>
                  </CardContent>
                </Card>
              ) : (
                results.map((candidate, index) => {
                  const overallScore = Number(candidate.overall_match || 0);
                  const semanticScore = Number(candidate.semantic_similarity || 0);
                  const tfidfScore = Number(candidate.tfidf_score || 0);
                  const strengthScore = Number(candidate.resume_strength_score || 0);
                  const expMatch = Number(candidate.experience_match_percentage || 0);
                  const skillMatch = Number(candidate.skill_match_percentage || 0);
                  const isActive = index === activeCandidateIndex;
                  const borderColor = isActive ? getScoreColor(overallScore) : "rgba(11, 27, 29, 0.12)";
                  const sectionScores = candidate.section_scores || {};
                  const roleFit = candidate.recommended_role || {};
                  const jobSkillsDetected = jobSkills.length > 0;
                  return (
                    <Card
                      key={`${candidate.file_name}-${index}`}
                      sx={{
                        ...panelCardSx,
                        border: `1.5px solid ${borderColor}`,
                        cursor: "pointer",
                        transition: "transform 160ms ease, border-color 160ms ease",
                        transform: isActive ? "scale(1.01)" : "scale(1)",
                      }}
                      className="result-rise"
                      onClick={() => setActiveCandidateIndex(index)}
                    >
                      <CardContent sx={{ p: { xs: 2.1, sm: 2.6 } }}>
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          justifyContent="space-between"
                          alignItems={{ xs: "flex-start", sm: "center" }}
                          spacing={1}
                          sx={{ mb: 1.5 }}
                        >
                          <Typography sx={{ color: "#173f45", fontSize: 23, fontWeight: 700 }}>
                            #{candidate.rank || index + 1} {candidate.candidate_name}
                          </Typography>
                          <Chip label={`${formatPercent(overallScore)}% Match`} sx={{ bgcolor: getScoreColor(overallScore), color: "white", fontWeight: 700 }} />
                        </Stack>

                        <MetricBar label="Overall Match" value={overallScore} />
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          spacing={1}
                          sx={{ mb: 1.4, "& > *": { flex: 1 } }}
                        >
                          <MetricBar label="Semantic Similarity" value={semanticScore} />
                          <MetricBar label="TF-IDF Similarity" value={tfidfScore} />
                        </Stack>
                        <Stack direction={{ xs: "column", sm: "row" }} spacing={1} useFlexGap flexWrap="wrap" sx={{ mb: 1.6 }}>
                          <Chip size="small" label={`Skill Match: ${skillMatch}%`} sx={{ bgcolor: "#dff7ef", color: "#1a6b5f", fontWeight: 700 }} />
                          <Chip size="small" label={`Resume Strength: ${strengthScore}%`} sx={{ bgcolor: "#e7f2f8", color: "#285f7c", fontWeight: 700 }} />
                          <Chip size="small" label={`Experience Match: ${expMatch}%`} sx={{ bgcolor: "#fff2d7", color: "#895e12", fontWeight: 700 }} />
                          {roleFit?.role && (
                            <Chip
                              size="small"
                              label={`Best suited: ${roleFit.role}${roleFit.confidence ? ` (${formatPercent(roleFit.confidence)}% conf.)` : ""}`}
                              sx={{ bgcolor: "rgba(33, 185, 123, 0.18)", color: "#127055", fontWeight: 700 }}
                            />
                          )}
                        </Stack>

                        {Object.keys(sectionScores).length > 0 && (
                          <Box sx={{ mb: 1.4 }}>
                            <Typography sx={{ color: "#1c4c53", fontWeight: 700, mb: 0.4 }}>Section Scores</Typography>
                            <Stack direction={{ xs: "column", sm: "row" }} spacing={0.8} useFlexGap flexWrap="wrap">
                              {Object.entries(sectionScores).map(([section, data]) => (
                                <Chip
                                  key={`${candidate.file_name}-${section}`}
                                  size="small"
                                  label={`${formatSectionLabel(section)}: ${formatPercent(data?.score || 0)}%`}
                                  sx={{ bgcolor: "rgba(11, 58, 63, 0.08)", color: "#1c5157", fontWeight: 600 }}
                                />
                              ))}
                            </Stack>
                          </Box>
                        )}

                        <Box sx={{ mb: 1.2 }}>
                          <Typography sx={{ color: "#1c4c53", fontWeight: 700, mb: 0.6 }}>Matched Skills</Typography>
                          <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap">
                            {(candidate.matched_skills || []).length > 0 ? (
                              candidate.matched_skills.map((skill) => (
                                <Chip key={skill} size="small" label={skill} sx={{ bgcolor: "#d8f5e8", color: "#186f60", fontWeight: 700 }} />
                              ))
                            ) : (
                              <Chip
                                size="small"
                                label={jobSkillsDetected ? "No matched skills" : "No JD skills detected"}
                                sx={{ bgcolor: "#edf1f2", color: "#607177" }}
                              />
                            )}
                          </Stack>
                        </Box>

                        <Box sx={{ mb: (candidate.suggestions || []).length > 0 ? 1.2 : 0 }}>
                          <Typography sx={{ color: "#1c4c53", fontWeight: 700, mb: 0.6 }}>Missing Skills</Typography>
                          <Stack direction="row" spacing={0.8} useFlexGap flexWrap="wrap">
                            {(candidate.missing_skills || []).length > 0 ? (
                              candidate.missing_skills.map((skill) => (
                                <Chip key={skill} size="small" label={skill} sx={{ bgcolor: "#fde3df", color: "#9f3a31", fontWeight: 700 }} />
                              ))
                            ) : (
                              <Chip
                                size="small"
                                label={jobSkillsDetected ? "All required skills present" : "Add skills to the JD to track gaps"}
                                sx={{ bgcolor: "#e0f6ec", color: "#1f785f", fontWeight: 700 }}
                              />
                            )}
                          </Stack>
                        </Box>

                        {(candidate.suggestions || []).length > 0 && (
                          <Box sx={{ p: 1.2, borderRadius: 2, bgcolor: "#eef9f5", border: "1px solid rgba(32, 109, 95, 0.17)" }}>
                            <Typography sx={{ color: "#1f5755", fontWeight: 700, mb: 0.45 }}>Suggestions</Typography>
                            {(candidate.suggestions || []).map((tip, tipIndex) => (
                              <Typography key={`${candidate.file_name}-${tipIndex}`} sx={{ color: "#3e6265", fontSize: 14.5 }}>
                                • {tip}
                              </Typography>
                            ))}
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  );
                })
              )}
            </Stack>
          </Grid>
        </Grid>
      </Container>

      <Backdrop open={isAnalyzing} sx={{ color: "#fff", zIndex: 2000, backdropFilter: "blur(4px)" }}>
        <Stack alignItems="center" spacing={2}>
          <CircularProgress color="inherit" size={56} />
          <Typography sx={{ fontWeight: 700, fontSize: 18 }}>Analyzing resumes...</Typography>
          <Typography sx={{ opacity: 0.86 }}>This may take a few moments.</Typography>
        </Stack>
      </Backdrop>
    </Box>
  );
}

export default App;
