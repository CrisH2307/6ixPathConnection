import { useMemo, useState, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Form,
  Button,
  Alert,
  ListGroup,
  Badge,
  Modal,
} from "react-bootstrap";
import { Routes, Route } from "react-router-dom";

const Home = () => {
  const [sourceName, setSourceName] = useState("");
  const [targetName, setTargetName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [path, setPath] = useState(100);
  const [showModal, setShowModal] = useState(false);
  const [selectedStep, setSelectedStep] = useState(null);
  const [routeId, setRouteId] = useState(null);
  const [hopsProgress, setHopsProgress] = useState([]);
  const [copied, setCopied] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/generate-routes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_name: sourceName,
          target_name: targetName,
          paths: path,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || data.message || "Request failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const firstPath = result?.paths && result.paths[0];
  const firstHop =
    firstPath && Array.isArray(firstPath.hops_detail)
      ? firstPath.hops_detail[0]
      : null;
  const signals = firstHop?.signals || {};
  //const categories_a = firstHop?.signals?.categories_a ?? [];
  //const categories_b = firstHop?.signals?.categories_b ?? [];
  const pathDetails = firstPath?.hops_detail || [];
  const allPaths = result?.paths ?? [];
  const outreachSequence = firstPath?.outreach_sequence || [];
  

  const nodeLookup = useMemo(() => {
    if (!firstPath?.nodes_detail) return {};
    return firstPath.nodes_detail.reduce((acc, node) => {
      if (node?._id) acc[node._id] = node;
      return acc;
    }, {});
  }, [firstPath]);

  const displayName = (nodeId) => {
    const node = nodeLookup[nodeId];
    return node?.name || node?.id || nodeId;
  };

  const describeSignals = (hopSignals = {}) => {
    const reasons = [];
    if (typeof hopSignals.skill_similarity === "number") {
      reasons.push(`Skill similarity ${hopSignals.skill_similarity.toFixed(2)}`);
    }
    if (hopSignals.same_school) reasons.push("Same school");
    if (hopSignals.same_company) reasons.push("Same company");
    const sharedCats = Array.isArray(hopSignals.categories_a)
      && Array.isArray(hopSignals.categories_b)
      ? hopSignals.categories_a.filter((cat) =>
          hopSignals.categories_b.includes(cat)
        )
      : [];
    if (sharedCats.length) {
      reasons.push(`Shared focus: ${sharedCats.join(", ")}`);
    }
    if (!reasons.length) reasons.push("No shared signals captured.");
    return reasons;
  };

  useEffect(() => {
    if (!result?.paths || !result.paths.length) {
      setRouteId(null);
      setHopsProgress([]);
      return;
    }

    const rid = buildRouteId(sourceName, targetName);
    const bestPath = result.paths[0];
    const hops = bestPath.hops_detail || [];

    if (!rid || !hops.length) {
      setRouteId(rid);
      setHopsProgress([]);
      return;
    }

    const defaultProgress = hops.map((hop, index) => ({
      step: index + 1,
      recipientId: hop.to,
      status: "not_started",
    }));

    const loaded = loadProgress(rid, defaultProgress);
      setRouteId(rid);
      setHopsProgress(loaded);
  }, [result, sourceName, targetName]);

  const handleOpenModal = (index) => {
    const stepData = outreachSequence[index];
    const hop = pathDetails[index];

    const senderName =
      result?.source_name ||
      (stepData?.sender_id ? displayName(stepData.sender_id) : sourceName || "You");

    const fallbackRecipientId = hop ? hop.to : null;
    const recipientName = stepData?.recipient_id
      ? displayName(stepData.recipient_id)
      : fallbackRecipientId
      ? displayName(fallbackRecipientId)
      : "Recipient";

    const warmIntroName = stepData?.warm_intro_id
      ? displayName(stepData.warm_intro_id)
      : null;

    const stepProgress = hopsProgress[index] || {
      step: index + 1,
      status: "not_started",
    };

    setSelectedStep({
      index,
      stepNumber: stepProgress.step ?? index + 1,
      senderName,
      recipientName,
      warmIntroName,
      message: stepData?.message || "",
      status: stepProgress.status || "not_started",
    });
    setCopied(false);
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setSelectedStep(null);
    setCopied(false); 
  };

  const handleSaveStep = () => {
    if (!selectedStep) return;

    const idx = selectedStep.index;

    setHopsProgress((prev) => {
      const next = [...prev];

      const existing = next[idx] || {
        step: idx + 1,
        recipientId: pathDetails[idx]?.to,
        status: "not_started",
      };

      next[idx] = {
        ...existing,
        status: selectedStep.status,
      };

      saveProgress(routeId, next);
      return next;
    });

    // Close modal after saving
    setShowModal(false);
    setSelectedStep(null);
  };


  const handleCopyMessage = async () => {
    if (!selectedStep?.message) return;
    try {
      await navigator.clipboard.writeText(selectedStep.message);
      setCopied(true);
      // optional: auto-reset after 2 seconds
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy message", err);
    }
  };

  const handleSelectedStatusChange = (newStatus) => {
    if (!selectedStep) return;
    setSelectedStep((prev) => ({ ...prev, status: newStatus }));
  };

  const STATUS_OPTIONS = [
    { value: "not_started", label: "Not started" },
    { value: "sent", label: "Sent" },
    { value: "waiting_reply", label: "Waiting reply" },
    { value: "done", label: "Done" },
    { value: "skipped", label: "Skipped" },
  ];

  const statusVariantMap = {
    not_started: "secondary",
    sent: "info",
    waiting_reply: "warning",
    done: "success",
    skipped: "dark",
  };

  const buildRouteId = (source, target) => {
    const s = (source || "").trim().toLowerCase();
    const t = (target || "").trim().toLowerCase();
    if (!s || !t) return null;
    return `${s}->${t}`;
  };

  const loadProgress = (routeId, defaultProgress) => {
    if (!routeId) return defaultProgress;
    try {
      const raw = window.localStorage.getItem(`route-progress:${routeId}`);
      if (!raw) return defaultProgress;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length === defaultProgress.length) {
        return parsed;
      }
    } catch (err) {
      console.error("Failed to load progress from localStorage", err);
    }
    return defaultProgress;
  };

  const saveProgress = (routeId, progress) => {
    if (!routeId) return;
    try {
      window.localStorage.setItem(
        `route-progress:${routeId}`,
        JSON.stringify(progress)
      );
    } catch (err) {
      console.error("Failed to save progress to localStorage", err);
    }
  };

  console.log(result);

  return (
    <>
    <Container className="py-4">
      <Row className="justify-content-center">
        <Col xs={12} md={12} lg={12} xl={10}>
          <h1 className="mb-4 text-center">6ixPathConnection</h1>

          {/* Form card */}
          <Card className="mb-4">
            <Card.Body>
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3">
                  <Form.Label>Source name</Form.Label>
                  <Form.Control
                    value={sourceName}
                    onChange={(e) => setSourceName(e.target.value)}
                    placeholder="e.g. Cris Huynh"
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Target name</Form.Label>
                  <Form.Control
                    value={targetName}
                    onChange={(e) => setTargetName(e.target.value)}
                    placeholder="e.g. Khoi Vu"
                  />
                </Form.Group>

                <div className="d-grid">
                  <Button type="submit" disabled={loading}>
                    {loading ? "Finding route..." : "Generate route"}
                  </Button>
                </div>
              </Form>

              {error && (
                <Alert variant="danger" className="mt-3 mb-0">
                  {error}
                </Alert>
              )}
            </Card.Body>
          </Card>

          {/* Result card */}
          {result && (
            <Card>
              <Card.Body>
                <Card.Title>Result</Card.Title>

                <p className="mb-2">
                  <strong>From:</strong> {result.source_name}
                  <br />
                  <strong>To:</strong> {result.target_name}
                </p>

                {pathDetails.length > 0 && (
                  <div className="mb-3">
                    <ListGroup>
                      {pathDetails.map((hop, index) => {
                        const reasons = describeSignals(hop.signals);
                        const progressForHop = hopsProgress[index] || { status: "not_started" };
                        const statusOption =
                          STATUS_OPTIONS.find((opt) => opt.value === progressForHop?.status) ||
                          STATUS_OPTIONS[0];
                        return (
                          <ListGroup.Item
                            key={`${hop.from}-${hop.to}-${index}`}
                            className="py-3"
                            action
                            onClick={() => handleOpenModal(index)}
                          >
                            <div className="d-flex align-items-center flex-wrap mb-2">
                              <span className="fw-bold me-2">Step {index + 1}</span>
                              <span className="me-2">{displayName(hop.from)}</span>
                              <span className="text-muted me-2">→</span>
                              <span>{displayName(hop.to)}</span>
                              {typeof hop.edge_weight === "number" && (
                                <Badge bg="info" className="ms-3 text-dark">
                                  Weight {hop.edge_weight.toFixed(2)}
                                </Badge>
                              )}
                              <Badge
                                bg={
                                  statusVariantMap[progressForHop?.status] ||
                                  statusVariantMap.not_started
                                }
                                className="ms-3"
                              >
                                {statusOption.label}
                              </Badge>
                            </div>
                            <div className="small text-muted">
                              {reasons.map((reason) => (
                                <Badge
                                  bg="light"
                                  key={`${hop.from}-${hop.to}-${index}-${reason}`}
                                  className="me-1 mb-1 text-dark"
                                >
                                  {reason}
                                </Badge>
                              ))}
                            </div>
                            <div className="mt-2 small text-muted">
                              Click to view message & update status
                            </div>
                          </ListGroup.Item>
                        );
                      })}
                    </ListGroup>
                  </div>
                )}

                {firstPath && (
                  <p className="mb-3">
                    <strong>Top path hops:</strong> {firstPath.hops}
                    <span className="ms-3">
                      <strong>Score:</strong>{" "}
                      {typeof firstPath.score === "number"
                        ? firstPath.score.toFixed(2)
                        : firstPath.score}
                    </span>
                  </p>
                )}

                {firstHop && (
                  <>
                    {/*
                    <p className="mb-2">
                      <strong>Skill similarity:</strong>{" "}
                      {typeof signals.skill_similarity === "number"
                        ? signals.skill_similarity.toFixed(2)
                        : signals.skill_similarity ?? "N/A"}
                    </p>
                    
                    <div className="mb-2">
                      {signals.same_school && (
                        <Badge bg="secondary" className="me-1">
                          Same school
                        </Badge>
                      )}
                      {signals.same_company && (
                        <Badge bg="secondary" className="me-1">
                          Same company
                        </Badge>
                      )}
                      {!signals.same_school && !signals.same_company && (
                        <Badge bg="secondary" className="me-1">
                          No shared school or company.
                        </Badge>
                      )}
                    </div>
                    */}
                    {/*
                    <div className="mb-2">
                      <strong>Categories_A:</strong>{" "}
                      {categories_a.length ? (
                        categories_a.map((cat) => (
                          <Badge key={cat} bg="secondary" className="me-1">
                            {cat}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-muted">None</span>
                      )}
                    </div>
                    
                    <div>
                      <strong>Categories_B:</strong>{" "}
                      {categories_b.length ? (
                        categories_b.map((cat) => (
                          <Badge key={cat} bg="secondary" className="me-1">
                            {cat}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-muted">None</span>
                      )}
                    </div>
                    */}
                  </>
                )}

                {!firstPath && (
                  <p className="mb-0">
                    {result.message || "No route found."}
                  </p>
                )}
              </Card.Body>
            </Card>
          )}

          {allPaths.length > 0 && (
            <Card className="mt-3">
              <Card.Body>
                <Card.Title>All paths</Card.Title>
                <ListGroup>
                  {allPaths.map((pathOption, idx) => {
                    const lookup = (pathOption.nodes_detail || []).reduce(
                      (acc, node) => {
                        if (node?._id) acc[node._id] = node;
                        return acc;
                      },
                      {}
                    );
                    const labels = pathOption.nodes.map(
                      (nodeId) =>
                        lookup[nodeId]?.name || lookup[nodeId]?.id || nodeId
                    );
                    return (
                      <ListGroup.Item key={`path-${idx}`} className="py-3">
                        <div className="d-flex justify-content-between flex-wrap">
                          <span className="fw-bold">Path {idx + 1}</span>
                          <span className="text-muted">
                            {pathOption.hops} hops · Score {" "}
                            {typeof pathOption.score === "number"
                              ? pathOption.score.toFixed(2)
                              : pathOption.score}
                          </span>
                        </div>
                        <div className="small text-muted mt-2">
                          {labels.map((label, labelIdx) => (
                            <span key={`${label}-${labelIdx}`}>
                              {label}
                              {labelIdx < labels.length - 1 && (
                                <span className="mx-1">→</span>
                              )}
                            </span>
                          ))}
                        </div>
                      </ListGroup.Item>
                    );
                  })}
                </ListGroup>
              </Card.Body>
            </Card>
          )}
        </Col>
      </Row>
    </Container>

    {/* Modal for outreach message */}
    <Modal show={showModal} onHide={handleCloseModal} centered size="lg">
      <Modal.Header closeButton>
        <Modal.Title>
          Outreach message – Step {selectedStep?.stepNumber}
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {selectedStep ? (
          <>
            <p className="mb-2">
              <strong>From:</strong> {selectedStep.senderName}
              <br />
              <strong>To:</strong> {selectedStep.recipientName}
            </p>
            {selectedStep.warmIntroName && (
              <p className="mb-2 text-muted">
                Mentions warm intro: {selectedStep.warmIntroName}
              </p>
            )}

            <Form.Group className="mb-3" controlId="statusSelect">
              <Form.Label className="fw-semibold">Step status</Form.Label>
              <Form.Select
                value={selectedStep.status}
                onChange={(e) => handleSelectedStatusChange(e.target.value)}
              >
                {STATUS_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </Form.Select>
            </Form.Group>

            <hr />
            <div className="d-flex justify-content-between align-items-center mb-2">
              <Form.Label className="fw-semibold mb-0">
                Suggested outreach message
              </Form.Label>

              <Button
                variant={"outline-secondary"}
                onClick={handleCopyMessage}
                disabled={!selectedStep?.message}
              >
                {copied ? "Copied" : "Copy message"}
              </Button>
            </div>

            <p style={{ whiteSpace: "pre-wrap" }}>{selectedStep.message}</p>

          </>
        ) : (
          <p className="mb-0 text-muted">
            No outreach message available for this step.
          </p>
        )}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleCloseModal}>
          Close
        </Button>
        <Button
          variant="primary"
          onClick={handleSaveStep}
          disabled={!selectedStep}
        >
          Save Status
        </Button>
      </Modal.Footer>
    </Modal>
    </>
  );
}

export default Home;
