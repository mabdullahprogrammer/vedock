document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.querySelector("#sidebar");
  document.querySelector("[data-sidebar-toggle]")?.addEventListener("click", () => sidebar?.classList.toggle("open"));

  document.querySelectorAll(".toast button").forEach((button) => button.addEventListener("click", () => button.closest(".toast")?.remove()));
  window.setTimeout(() => document.querySelectorAll(".toast").forEach((toast) => toast.remove()), 8000);

  const modelDrawer = document.querySelector("[data-model-drawer]");
  document.querySelector("[data-model-drawer-toggle]")?.addEventListener("click", () => modelDrawer?.classList.toggle("closed"));
  document.querySelector("[data-model-drawer-close]")?.addEventListener("click", () => modelDrawer?.classList.add("closed"));
  document.querySelector("[data-show-history]")?.addEventListener("click", () => {
    document.querySelector("[data-drawer-overview]")?.setAttribute("hidden", "");
    document.querySelector("[data-drawer-history]")?.removeAttribute("hidden");
  });
  document.querySelector("[data-hide-history]")?.addEventListener("click", () => {
    document.querySelector("[data-drawer-history]")?.setAttribute("hidden", "");
    document.querySelector("[data-drawer-overview]")?.removeAttribute("hidden");
  });
  const customizeDialog = document.querySelector("[data-customize-dialog]");
  document.querySelector("[data-customize-open]")?.addEventListener("click", () => customizeDialog?.showModal());
  document.querySelector("[data-customize-close]")?.addEventListener("click", () => customizeDialog?.close());
  customizeDialog?.addEventListener("click", (event) => {
    if (event.target === customizeDialog) customizeDialog.close();
  });

  document.querySelectorAll(".task-card input[type='radio']").forEach((radio) => {
    const updateCards = () => {
      document.querySelectorAll(`input[name='${radio.name}']`).forEach((item) => item.closest(".task-card")?.classList.toggle("active", item.checked));
    };
    radio.addEventListener("change", updateCards);
    updateCards();
  });

  document.querySelectorAll("[data-wizard]").forEach((wizard) => {
    const steps = [...wizard.querySelectorAll("[data-wizard-step]")];
    const rail = document.querySelector("[data-wizard-rail]");
    const railItems = rail ? [...rail.children] : [];
    const previous = wizard.querySelector("[data-wizard-prev]");
    const next = wizard.querySelector("[data-wizard-next]");
    const label = wizard.querySelector("[data-wizard-label]");
    let index = 0;
    const available = (direction = 1) => {
      let candidate = index + direction;
      while (candidate >= 0 && candidate < steps.length && steps[candidate].hidden) candidate += direction;
      return candidate;
    };
    const show = () => {
      if (steps[index]?.hidden) {
        const candidate = available(1);
        if (candidate < steps.length) index = candidate;
      }
      steps.forEach((step, position) => step.classList.toggle("wizard-current", position === index));
      railItems.forEach((item, position) => {
        item.classList.toggle("active", position === index);
        item.classList.toggle("complete", position < index);
      });
      if (previous) previous.hidden = available(-1) < 0;
      if (next) next.hidden = available(1) >= steps.length;
      if (label) label.textContent = `Step ${index + 1} of ${steps.length}`;
      steps[index]?.scrollIntoView({behavior: "smooth", block: "start"});
    };
    next?.addEventListener("click", () => {
      const active = steps[index];
      const invalid = [...active.querySelectorAll("input,select,textarea")].find((control) => !control.disabled && !control.checkValidity());
      if (invalid) { invalid.reportValidity(); invalid.focus(); return; }
      const candidate = available(1);
      if (candidate < steps.length) { index = candidate; show(); }
    });
    previous?.addEventListener("click", () => {
      const candidate = available(-1);
      if (candidate >= 0) { index = candidate; show(); }
    });
    document.addEventListener("wizard-refresh", show);
    show();
  });

  const studio = document.querySelector("[data-model-studio]");
  if (studio) {
    const sourceSelect = studio.querySelector("[data-model-source]");
    const buildModeControls = [...studio.querySelectorAll("input[name='build_mode']")];
    const sourceSection = studio.querySelector("[data-source-section]");
    const trainingSection = studio.querySelector("[data-training-section]");
    const parameterSection = studio.querySelector("[data-parameter-section]");
    const datasetSelect = studio.querySelector("[data-dataset-select]");
    const finetuneMethod = studio.querySelector("[data-finetune-method]");
    const submit = studio.querySelector("[data-studio-submit]");
    const scratchPreset = studio.querySelector("[data-scratch-preset]");
    const customArchitecture = studio.querySelector("[data-custom-architecture]");

    const setSectionEnabled = (section, enabled) => {
      if (!section) return;
      section.hidden = !enabled;
      section.querySelectorAll("input,select,textarea,button").forEach((control) => { control.disabled = !enabled; });
    };
    const updateSource = () => {
      if (!sourceSelect) return;
      studio.querySelectorAll("[data-source-fields]").forEach((group) => {
        const enabled = group.dataset.sourceFields === sourceSelect.value;
        group.hidden = !enabled;
        group.querySelectorAll("input,select,textarea").forEach((control) => { control.disabled = !enabled; });
      });
      if (customArchitecture) {
        const custom = sourceSelect.value === "scratch" && scratchPreset?.value === "custom";
        customArchitecture.hidden = !custom;
        customArchitecture.querySelectorAll("input").forEach((control) => { control.disabled = !custom; });
      }
    };
    const updateMode = () => {
      const mode = buildModeControls.find((control) => control.checked)?.value || "fine_tune";
      if (mode === "scratch" && sourceSelect?.querySelector("option[value='scratch']")) sourceSelect.value = "scratch";
      const merge = mode === "merge";
      const inferenceOnly = mode === "inference_only";
      setSectionEnabled(sourceSection, !merge);
      setSectionEnabled(trainingSection, !merge && !inferenceOnly);
      setSectionEnabled(parameterSection, !merge && !inferenceOnly);
      if (datasetSelect) datasetSelect.required = !merge && !inferenceOnly;
      if (finetuneMethod) finetuneMethod.hidden = mode !== "fine_tune";
      if (submit) submit.textContent = merge ? "Open merge compatibility" : (inferenceOnly ? "Register model" : "Save project and review");
      updateSource();
      document.dispatchEvent(new CustomEvent("wizard-refresh"));
    };
    sourceSelect?.addEventListener("change", () => {
      if (sourceSelect.value !== "scratch") {
        const scratchMode = buildModeControls.find((control) => control.value === "scratch" && control.checked);
        const fineTuneMode = buildModeControls.find((control) => control.value === "fine_tune");
        if (scratchMode && fineTuneMode) fineTuneMode.checked = true;
      }
      updateMode();
    });
    scratchPreset?.addEventListener("change", updateSource);
    buildModeControls.forEach((control) => control.addEventListener("change", updateMode));
    updateMode();
  }

  document.querySelectorAll("[data-advanced-toggle]").forEach((button) => button.addEventListener("click", () => {
    const panel = document.querySelector("[data-advanced-panel]");
    if (panel) panel.hidden = !panel.hidden;
  }));
  document.querySelectorAll("[data-advanced-toggle-checkbox]").forEach((input) => input.addEventListener("change", () => {
    const scope = input.closest(".model-chat-shell") || input.closest("form");
    scope?.classList.toggle("show-advanced", input.checked);
  }));

  document.querySelectorAll("[data-depends]").forEach((field) => {
    const dependencies = JSON.parse(field.dataset.depends || "{}");
    const update = () => {
      const enabled = Object.entries(dependencies).every(([name, expected]) => {
        const control = field.closest("form")?.querySelector(`[name='${name}']`) || document.querySelector(`[name='${name}']`);
        if (!control) return false;
        const value = control.type === "checkbox" ? control.checked : control.value;
        return String(value) === String(expected);
      });
      field.hidden = !enabled;
      field.querySelectorAll("input,select,textarea").forEach((control) => { control.disabled = !enabled; });
    };
    Object.keys(dependencies).forEach((name) => (field.closest("form")?.querySelector(`[name='${name}']`) || document.querySelector(`[name='${name}']`))?.addEventListener("change", update));
    update();
  });

  document.querySelector("[data-model-switch]")?.addEventListener("change", (event) => { window.location.href = event.target.value; });

  document.querySelector("[data-model-search]")?.addEventListener("input", (event) => {
    const query = event.target.value.trim().toLowerCase();
    document.querySelectorAll("[data-model-card]").forEach((card) => {
      card.hidden = Boolean(query) && !String(card.dataset.search || "").toLowerCase().includes(query);
    });
  });

  document.querySelectorAll("[data-api-try]").forEach((button) => button.addEventListener("click", async () => {
    const output = document.querySelector("[data-api-output]");
    button.disabled = true;
    if (output) output.textContent = "Loading...";
    try {
      const response = await fetch(button.dataset.apiTry, {headers: {"Accept": "application/json"}});
      const payload = await response.json();
      if (output) output.textContent = JSON.stringify(payload, null, 2);
    } catch (error) {
      if (output) output.textContent = `Request failed: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  }));

  const prompt = document.querySelector("textarea[name='prompt']");
  const count = document.querySelector("[data-character-count]");
  if (prompt && count) {
    const update = () => { count.textContent = prompt.value.length.toLocaleString(); };
    prompt.addEventListener("input", update); update();
  }

  const schemaSelect = document.querySelector("[data-schema-select]");
  if (schemaSelect) {
    const fieldSets = {
      prompt_response: ["prompt", "response"], text_completion: ["text"],
      instruction: ["instruction", "input", "output"], chat: ["system", "prompt", "response"],
      classification: ["text", "label"], image_classification: ["image", "label"],
      tabular_supervised: ["features", "target"]
    };
    const update = () => document.querySelectorAll("[data-map-field]").forEach((field) => {
      field.hidden = !fieldSets[schemaSelect.value].includes(field.dataset.mapField);
    });
    schemaSelect.addEventListener("change", update); update();
  }

  const datasetRecipes = {
    quality: ["trim_whitespace", "normalize_unicode", "remove_control_characters", "remove_empty_records", "remove_duplicates"],
    web: ["trim_whitespace", "normalize_unicode", "remove_html", "remove_urls", "remove_control_characters", "remove_empty_records", "remove_duplicates"],
    privacy: ["trim_whitespace", "normalize_unicode", "redact_emails", "redact_phone_numbers", "remove_control_characters", "remove_empty_records", "remove_duplicates"],
    minimal: ["trim_whitespace", "remove_empty_records"],
  };
  document.querySelectorAll("[data-dataset-recipe]").forEach((button) => button.addEventListener("click", () => {
    const selected = new Set(datasetRecipes[button.dataset.datasetRecipe] || []);
    document.querySelectorAll(".compact-clean input[type='checkbox']").forEach((input) => { input.checked = selected.has(input.name); });
    document.querySelectorAll("[data-dataset-recipe]").forEach((item) => item.classList.toggle("active", item === button));
  }));
  const applyRecommendations = document.querySelector("[data-apply-recommendations]");
  if (applyRecommendations) applyRecommendations.addEventListener("click", () => {
    const source = document.querySelector("[data-recommended-operations]");
    let operations = [];
    try { operations = JSON.parse(source?.textContent || "[]"); } catch (_) { operations = []; }
    document.querySelectorAll("[data-clean-operation]").forEach((input) => {
      if (operations.includes(input.dataset.cleanOperation)) input.checked = true;
    });
    applyRecommendations.textContent = operations.length ? `${operations.length} recommendations applied` : "Nothing automatic to apply";
    document.querySelector('[data-wizard-next]')?.focus();
  });

  const presets = {
    fast: {max_steps: 1, max_examples: 20, max_seq_length: 64, per_device_train_batch_size: 1, gradient_accumulation_steps: 1},
    balanced: {max_steps: 10, max_examples: 200, max_seq_length: 128, per_device_train_batch_size: 1, gradient_accumulation_steps: 2},
    quality: {max_steps: 100, max_examples: 2000, max_seq_length: 256, per_device_train_batch_size: 1, gradient_accumulation_steps: 4},
    memory: {max_steps: 5, max_examples: 100, max_seq_length: 64, per_device_train_batch_size: 1, gradient_accumulation_steps: 4, lora_r: 4}
  };
  document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll("[data-preset]").forEach((item) => item.classList.remove("active")); button.classList.add("active");
    const values = presets[button.dataset.preset] || {};
    Object.entries(values).forEach(([name, value]) => { const input = document.querySelector(`[name='${name}']`); if (input) input.value = value; });
  }));

  const autoRefresh = document.querySelector("[data-auto-refresh]");
  if (autoRefresh) window.setTimeout(() => window.location.reload(), Number(autoRefresh.dataset.autoRefresh || 4000));

  const chatForm = document.querySelector("[data-chat-form]");
  if (chatForm) {
    const payload = chatForm.querySelector(".chat-parameter-payload");
    const visibleParameters = document.querySelector("[data-visible-parameters]");
    if (payload && visibleParameters) {
      payload.querySelectorAll("input,select,textarea,button").forEach((control) => control.setAttribute("form", chatForm.id));
      while (payload.firstChild) visibleParameters.appendChild(payload.firstChild);
      payload.remove();
    }
    const textarea = chatForm.querySelector("textarea[name='prompt']");
    const thread = document.querySelector("[data-chat-thread]");
    const status = chatForm.querySelector("[data-chat-status]");
    const sendButton = chatForm.querySelector("[data-send-button]");
    const stopButton = chatForm.querySelector("[data-stop-button]");
    const conversationInput = chatForm.querySelector("[data-conversation-id]");
    const modelLoadStatus = document.querySelector("[data-model-load-status]");
    let activeGenerationId = null;
    const appendMessage = (role, label, content = "") => {
      thread?.querySelector(".chat-empty")?.remove();
      const article = document.createElement("article");
      article.className = role;
      const name = document.createElement("span"); name.textContent = label;
      const body = document.createElement("p"); body.textContent = content;
      article.append(name, body); thread?.appendChild(article);
      thread?.scrollTo({top: thread.scrollHeight, behavior: "smooth"});
      return body;
    };
    const submitChat = async () => {
      const promptValue = textarea?.value.trim();
      if (!promptValue || sendButton?.disabled) return;
      const streaming = document.querySelector("[name='streaming']")?.checked;
      if (!streaming) {
        chatForm.action = chatForm.dataset.fallbackAction;
        chatForm.submit();
        return;
      }
      const modelName = document.querySelector(".chat-model-header h1")?.textContent || "Model";
      appendMessage("user", "You", promptValue);
      const assistantBody = appendMessage("assistant streaming", modelName, "");
      textarea.value = "";
      sendButton.disabled = true;
      if (stopButton) stopButton.hidden = false;
      status.textContent = "Generating...";
      if (modelLoadStatus) modelLoadStatus.textContent = "loading / generating";
      try {
        const parameters = {};
        const chatSettings = {};
        const chatSettingNames = new Set(["use_history", "context_override", "context_limit"]);
        [...chatForm.elements].forEach((control) => {
          if (!control.name || ["prompt", "conversation_id", "csrf_token"].includes(control.name) || control.disabled) return;
          const value = control.type === "checkbox" ? control.checked : control.value;
          if (chatSettingNames.has(control.name)) chatSettings[control.name] = value;
          else parameters[control.name] = value;
        });
        const csrfToken = chatForm.querySelector("[name='csrf_token']")?.value || "";
        const response = await fetch(chatForm.action, {
          method: "POST",
          body: JSON.stringify({prompt: promptValue, conversation_id: conversationInput.value, parameters, ...chatSettings}),
          headers: {"Accept": "text/event-stream", "Content-Type": "application/json", "X-CSRF-Token": csrfToken}
        });
        if (!response.ok) {
          const failure = await response.json().catch(() => ({}));
          throw new Error(failure.error?.message || `Request failed with status ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let eventName = "message";
        while (true) {
          const {value, done} = await reader.read();
          buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";
          for (const block of blocks) {
            eventName = "message";
            let data = "";
            block.split("\n").forEach((line) => {
              if (line.startsWith("event:")) eventName = line.slice(6).trim();
              if (line.startsWith("data:")) data += line.slice(5).trim();
            });
            if (!data) continue;
            const parsed = JSON.parse(data);
            if (eventName === "started") {
              activeGenerationId = parsed.generation_id;
            } else if (eventName === "message" && parsed.token !== undefined) {
              assistantBody.textContent += parsed.token;
              thread?.scrollTo({top: thread.scrollHeight});
            } else if (eventName === "done") {
              conversationInput.value = parsed.conversation_id;
              const url = new URL(window.location.href);
              url.searchParams.set("conversation", parsed.conversation_id);
              history.replaceState({}, "", url);
              activeGenerationId = null;
            } else if (eventName === "error") {
              throw new Error(parsed.message || "Generation failed");
            }
          }
          if (done) break;
        }
        assistantBody.closest("article")?.classList.remove("streaming");
        status.textContent = "Ready";
        if (modelLoadStatus) modelLoadStatus.textContent = "loaded in memory";
      } catch (error) {
        assistantBody.textContent = `Error: ${error.message}`;
        assistantBody.closest("article")?.classList.add("error");
        status.textContent = "The runtime returned an error. You can adjust parameters and retry.";
        if (modelLoadStatus) modelLoadStatus.textContent = "runtime error - retry available";
      } finally {
        sendButton.disabled = false;
        if (stopButton) stopButton.hidden = true;
        activeGenerationId = null;
        textarea?.focus();
      }
    };
    stopButton?.addEventListener("click", async () => {
      if (!activeGenerationId) return;
      stopButton.disabled = true;
      status.textContent = "Stopping after the current token...";
      try {
        const csrfToken = chatForm.querySelector("[name='csrf_token']")?.value || "";
        await fetch(chatForm.dataset.stopAction, {
          method: "POST",
          body: JSON.stringify({generation_id: activeGenerationId}),
          headers: {"Content-Type": "application/json", "X-CSRF-Token": csrfToken}
        });
      } finally {
        stopButton.disabled = false;
      }
    });
    chatForm.addEventListener("submit", (event) => { event.preventDefault(); submitChat(); });
    textarea?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitChat(); }
    });
  }
});
