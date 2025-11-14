import { useState } from "react";
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
                        <Badge key={signals.same_school} bg="secondary" className="me-1">
                          Same school
                        </Badge>
                      )}
                      {signals.same_company && (
                        <Badge key={signals.same_company} bg="secondary" className="me-1">
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
