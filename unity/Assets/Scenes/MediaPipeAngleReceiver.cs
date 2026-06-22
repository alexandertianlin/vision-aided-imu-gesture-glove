using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// V2.2: Receives continuous per-finger angle packets from MediaPipe (port 5058)
/// and drives FingerSolver directly. Falls back to IMU if packets timeout.
/// Attach to same GameObject as HandMotionManager.
/// 
/// Packet format: {"ts":int, "seq":int, "type":"ang",
///   "fi":int(0-4), "p":float(pitch), "y":float(yaw),
///   "ja":[float,float], "c":float(confidence)}
/// </summary>
[RequireComponent(typeof(HandMotionManager))]
public class MediaPipeAngleReceiver : MonoBehaviour
{
    [Header("Connection")]
    public int listenPort = 5058;
    public bool enableAngleDriving = true;

    [Header("Timeout")]
    [Tooltip("Seconds without a packet before IMU resumes full control.")]
    public float angleTimeoutSeconds = 0.5f;

    [Header("Blending")]
    [Range(0.1f, 30f)] public float moveSpeed = 18f;

    [Header("Debug")]
    public bool printLog = true;

    private HandMotionManager handMotion;
    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool running;

    private readonly object angleLock = new object();
    private float[] lastPitches = new float[5];
    private float[] lastYaws = new float[5];
    private float[] lastConfidences = new float[5];
    private float[] lastPacketTimes = new float[5];
    private bool[] hasReceived = new bool[5];

    private void Awake()
    {
        handMotion = GetComponent<HandMotionManager>();
        for (int i = 0; i < 5; i++)
        {
            lastPacketTimes[i] = -999f;
            lastConfidences[i] = 0f;
        }
    }

    private void OnEnable() { StartReceiver(); }
    private void OnDisable() { StopReceiver(); }

    private void LateUpdate()
    {
        if (!enableAngleDriving || handMotion == null || !handMotion.IsCalibrated)
            return;

        float now = Time.time;

        for (int fi = 0; fi < 5; fi++)
        {
            if (!hasReceived[fi]) continue;
            if (now - lastPacketTimes[fi] > angleTimeoutSeconds)
            {
                if (printLog && hasReceived[fi])
                    Debug.Log($"[AngleRecv] fi={fi} TIMEOUT - IMU resumes control");
                hasReceived[fi] = false;
                continue;
            }

            float pitch, yaw;
            lock (angleLock)
            {
                pitch = lastPitches[fi];
                yaw = lastYaws[fi];
            }

            // Apply via FingerSolver.ForceVisionAngleAnchor
            var fingers = GetFingers();
            if (fi < fingers.Length && fingers[fi] != null)
            {
                float blend = 1f - Mathf.Exp(-moveSpeed * Time.deltaTime);
                fingers[fi].MoveVisionAngleAnchorAndSyncImu(
                    pitch, yaw, blend,
                    handMotion.fingerBendAxis,
                    handMotion.fingerSpreadAxis
                );
            }
        }
    }

    private FingerSolver[] GetFingers()
    {
        return new FingerSolver[] {
            handMotion.thumb, handMotion.index, handMotion.middle,
            handMotion.ring, handMotion.little
        };
    }

    private void StartReceiver()
    {
        if (udpClient != null) return;
        try
        {
            udpClient = new UdpClient(listenPort);
            udpClient.Client.ReceiveTimeout = 200;
            running = true;
            receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
            receiveThread.Start();
            Debug.Log($"[AngleRecv] Started on 0.0.0.0:{listenPort}");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[AngleRecv] Start failed: {e.Message}");
            StopReceiver();
        }
    }

    private void StopReceiver()
    {
        running = false;
        if (udpClient != null) { udpClient.Close(); udpClient = null; }
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Join(300);
        receiveThread = null;
    }

    [Serializable]
    private class AnglePacket
    {
        public long ts;
        public int seq;
        public string type;
        public int fi;
        public float p;
        public float y;
        public List<float> ja;
        public float c;
    }

    private void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (running && udpClient != null)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remote);
                string json = Encoding.UTF8.GetString(data);
                var pkt = JsonUtility.FromJson<AnglePacket>(json);
                if (pkt == null || pkt.type != "ang") continue;
                if (pkt.fi < 0 || pkt.fi > 4) continue;

                lock (angleLock)
                {
                    lastPitches[pkt.fi] = pkt.p;
                    lastYaws[pkt.fi] = pkt.y;
                    lastConfidences[pkt.fi] = pkt.c;
                    lastPacketTimes[pkt.fi] = Time.time;
                    hasReceived[pkt.fi] = true;
                }

                if (printLog)
                    Debug.Log($"[AngleRecv] fi={pkt.fi} p={pkt.p:F1} y={pkt.y:F1} ja=[{string.Join(",", pkt.ja)}] c={pkt.c:F3}");
            }
            catch (SocketException) { }
            catch (ObjectDisposedException) { break; }
            catch (Exception e)
            {
                Debug.LogWarning($"[AngleRecv] Error: {e.Message}");
            }
        }
    }
}
