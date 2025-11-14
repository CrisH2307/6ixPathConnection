import { useMemo, useState } from "react";
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
} from "react-bootstrap";

function App() {
  const [sourceName, setSourceName] = useState("");
  const [targetName, setTargetName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [path, setPath] = useState(100);

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
  const categories_a = firstHop?.signals?.categories_a ?? [];
  const categories_b = firstHop?.signals?.categories_b ?? [];
  const pathDetails = firstPath?.hops_detail || [];
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

  console.log(result);

  return (
    <Container className="py-4">
      <Row className="justify-content-center">
        <Col md={8} lg={6}>
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
                        return (
                          <ListGroup.Item
                            key={`${hop.from}-${hop.to}-${index}`}
                            className="py-3"
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
        </Col>
      </Row>
    </Container>
  );
}

export default App;
