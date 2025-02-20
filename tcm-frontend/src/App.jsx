import { useState } from 'react';
import './App.css'; // 引入 CSS 文件

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [externalDb, setExternalDb] = useState(true); // 是否使用知识库，默认为 true
  const [model, setModel] = useState("Qwen/Qwen2.5-0.5B-Instruct"); // 选择的模型
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    try {
      if (!question.trim()) return;

      setLoading(true);
      setMessages([...messages, { text: question, sender: "user" }]);
      const response = await fetch("http://localhost:5001/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          question: question, 
          model,
          external_db: externalDb, // 传递 external_db 字段
        }),
      });
    
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let generatedText = "";
      let printable = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
       
        if(chunk.indexOf("Answer:") >-1) {
          printable = true;
          generatedText += chunk.split('Answer:')[1]
        } else {
          if (printable) {
            generatedText += chunk;
          }
        }
        // setAnswer(() => 
        //   // 如果结尾是 <|endoftext|>， 则去除
        //   chunk.endsWith("<|endoftext|>") ? 
        //     chunk.slice(0, -"<|endoftext|>".length) : 
        //     chunk
        // ); // 逐步更新响应内容
        setMessages([...messages, {text: generatedText, sender: "bot" }]);
      }
      setLoading(false);
      setQuestion("");
    } catch (error) {
      console.error("Error fetching answer:", error);
    }
  };

  return (
    <div className="app-container">
      {/* 侧边栏 */}
      <div className="sidebar">
        <div>
          <h2>TCM AI</h2>
          <button className="sidebar-button">新对话</button>
          <button className="sidebar-button">历史对话</button>
          <button className="sidebar-button">最近问题</button>
        </div>
        <div>
          <button className="sidebar-button">设置</button>
          <button className="sidebar-button">退出</button>
        </div>
      </div>

      {/* 主内容区 */}
      <div className="main-content">
        {/* 答案区域 */}
        <div className="answer-section">
          <h2>答案：</h2>
          <p className="answer-text">
            {
            messages.map((msg, index) => (
              <p key={index} style={{ color: msg.sender === "user" ? "blue" : "green" }}>
                {msg.text}
              </p>
            ))
            }
          </p>
        </div>

        {/* 问题输入区域 */}
        <div className="question-section">
          <textarea
            rows="4"
            cols="50"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="请输入问题..."
            className="question-textarea"
          />
          <div style={{ textAlign: "center", marginBottom: "20px" }}>
            <button onClick={askQuestion} className="submit-button" disabled={loading}>
              {loading ? "生成中..." : "发送"}
            </button>
          </div>

          {/* 单选框区域 */}
          <div className="radio-section">
            <label className="radio-label">
              <input
                type="radio"
                name="external_db"
                value="true"
                checked={externalDb === true}
                onChange={() => setExternalDb(true)}
                style={{ marginRight: "5px" }}
              />
              使用知识库
            </label>
            <label>
              <input
                type="radio"
                name="external_db"
                value="false"
                checked={externalDb === false}
                onChange={() => setExternalDb(false)}
                style={{ marginRight: "5px" }}
              />
              不使用知识库
            </label>
          </div>
          {/* 选择模型 */}
          <div style={{ marginBottom: "20px", textAlign: "center" }}>
            <label style={{ marginRight: "10px" }}>选择模型：</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="model-select"
            >
              <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5-0.5B</option>
              <option value="Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5-1.5B</option>
              <option value="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">DeepSeek-Qwen-1.5B</option>
              <option value="/Users/pony/work/own/llm/langchain/traditional_chinese_medicine_llm_app/fine_tuning/Qwen/tcm_finetuned_qwen">Qwen2.5-0.5B-fine_tuned</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;