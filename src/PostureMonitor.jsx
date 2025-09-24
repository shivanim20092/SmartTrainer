import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";

export default function PostureMonitor() {
  const [status, setStatus] = useState("⏳ Loading model...");
  const [detector, setDetector] = useState(null);
  const [feedback, setFeedback] = useState([]);
  const [score, setScore] = useState(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Load MoveNet model
  useEffect(() => {
    async function loadModel() {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const detector = await poseDetection.createDetector(
          poseDetection.SupportedModels.MoveNet,
          {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          }
        );

        setDetector(detector);
        setStatus("✅ Model ready");
      } catch (err) {
        console.error("Model failed to load", err);
        setStatus("❌ Model failed: " + err.message);
      }
    }
    loadModel();
  }, []);

  // Handle image upload
  async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file || !detector) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
      imageRef.current = img;

      const poses = await detector.estimatePoses(img);
      analyzePosture(poses);
      drawResult(img, poses);
    };
  }

  // Draw skeleton and keypoints
  function drawResult(img, poses, issues = {}) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = img.width;
    canvas.height = img.height;

    ctx.drawImage(img, 0, 0, img.width, img.height);

    if (poses.length > 0) {
      const keypoints = poses[0].keypoints;
      ctx.fillStyle = "red";
      ctx.lineWidth = 3;

      keypoints.forEach((pt) => {
        if (pt.score > 0.5) {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
          ctx.fill();
        }
      });

      const leftShoulder = keypoints.find((k) => k.name === "left_shoulder");
      const rightShoulder = keypoints.find((k) => k.name === "right_shoulder");
      const leftHip = keypoints.find((k) => k.name === "left_hip");
      const rightHip = keypoints.find((k) => k.name === "right_hip");

      if (leftShoulder && rightShoulder) {
        ctx.strokeStyle = issues.shoulders ? "red" : "lime";
        ctx.beginPath();
        ctx.moveTo(leftShoulder.x, leftShoulder.y);
        ctx.lineTo(rightShoulder.x, rightShoulder.y);
        ctx.stroke();
      }

      if (leftHip && rightHip) {
        ctx.strokeStyle = issues.hips ? "red" : "lime";
        ctx.beginPath();
        ctx.moveTo(leftHip.x, leftHip.y);
        ctx.lineTo(rightHip.x, rightHip.y);
        ctx.stroke();
      }
    }
  }

  // Analyze posture & calculate score
  function analyzePosture(poses) {
    if (poses.length === 0) {
      setFeedback(["No person detected."]);
      setScore(null);
      return;
    }

    const keypoints = poses[0].keypoints;
    const leftShoulder = keypoints.find((k) => k.name === "left_shoulder");
    const rightShoulder = keypoints.find((k) => k.name === "right_shoulder");
    const nose = keypoints.find((k) => k.name === "nose");
    const leftHip = keypoints.find((k) => k.name === "left_hip");
    const rightHip = keypoints.find((k) => k.name === "right_hip");

    let messages = [];
    let issues = { shoulders: false, head: false, hips: false };
    let deductions = 0;

    if (leftShoulder && rightShoulder) {
      const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
      if (shoulderDiff > 30) {
        issues.shoulders = true;
        deductions += 20;
        messages.push(
          "⚠️ **Uneven shoulders** — may cause muscle imbalance.\n💡 *Correction:* Adjust chair or desk height."
        );
      }
    }

    if (nose && leftShoulder && rightShoulder) {
      const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      if (nose.y < avgShoulderY - 80) {
        issues.head = true;
        deductions += 30;
        messages.push(
          "⚠️ **Forward head posture** — increases cervical spine stress.\n💡 *Correction:* Raise monitor to eye level."
        );
      }
    }

    if (leftHip && rightHip) {
      const hipDiff = Math.abs(leftHip.y - rightHip.y);
      if (hipDiff > 30) {
        issues.hips = true;
        deductions += 20;
        messages.push(
          "⚠️ **Uneven hips** — possible pelvic tilt.\n💡 *Correction:* Sit evenly, keep both feet flat."
        );
      }
    }

    if (messages.length === 0) {
      messages.push("✅ Posture looks good — keep it up!");
    }

    const finalScore = Math.max(100 - deductions, 40); // minimum 40
    setScore(finalScore);
    setFeedback(messages);

    if (imageRef.current) {
      drawResult(imageRef.current, poses, issues);
    }
  }

  // Color score
  function scoreColor(score) {
    if (score >= 85) return "green";
    if (score >= 70) return "orange";
    return "red";
  }

  return (
    <div style={{ padding: "1rem", fontFamily: "sans-serif" }}>
      <h1>🧍 Posture Monitor</h1>
      <p>
        <strong>Model status:</strong> {status}
      </p>

      <input type="file" accept="image/*" onChange={handleImageUpload} />

      <div style={{ marginTop: "1rem" }}>
        <canvas
          ref={canvasRef}
          style={{ border: "1px solid #ccc", maxWidth: "100%" }}
        />
      </div>

      {score !== null && (
        <div
          style={{
            marginTop: "1rem",
            padding: "1rem",
            borderRadius: "10px",
            backgroundColor: "#f9f9f9",
            boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
          }}
        >
          <h2 style={{ color: scoreColor(score) }}>
            📊 Posture Score: {score}/100
          </h2>
          <ul>
            {feedback.map((msg, i) => (
              <li
                key={i}
                style={{ marginBottom: "0.5rem" }}
                dangerouslySetInnerHTML={{
                  __html: msg.replace(/\n/g, "<br/>"),
                }}
              />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
