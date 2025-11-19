import { useState } from "react";
import { Container, Card, Form, Button, Alert, ListGroup } from "react-bootstrap";

const SimilarPeople = () => {
  const [sourceName, setSourceName] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchSimilar = async (e) => {
    e.preventDefault();
    if (!sourceName.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch("http://localhost:8000/similar-people", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_name: sourceName.trim(), top_k: topK }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setResults(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="py-4">
      <Card className="mb-4">
        <Card.Body>
          <Form onSubmit={fetchSimilar}>
            <Form.Group className="mb-3">
              <Form.Label>Person name</Form.Label>
              <Form.Control
                value={sourceName}
                onChange={(e) => setSourceName(e.target.value)}
                placeholder="e.g. Cris Huynh"
              />
            </Form.Group>
            <Form.Group className="mb-3">
              <Form.Label>Top K</Form.Label>
              <Form.Control
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </Form.Group>
            <Button type="submit" disabled={loading}>
              {loading ? "Searching..." : "Find similar people"}
            </Button>
          </Form>
          {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
        </Card.Body>
      </Card>

      {results.length > 0 && (
        <Card>
          <Card.Body>
            <Card.Title>Similar people</Card.Title>
            <ListGroup>
              {results.map((r, idx) => (
                <ListGroup.Item key={`${r.name}-${idx}`}>
                  <strong>{r.name}</strong> — {r.role} @ {r.company}
                  <span className="ms-2 text-muted">Score: {r.score.toFixed(3)}</span>
                </ListGroup.Item>
              ))}
            </ListGroup>
          </Card.Body>
        </Card>
      )}
    </Container>
  );
};

export default SimilarPeople;
