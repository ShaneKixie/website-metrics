import { registerAccountTools } from "./accountTools.js";
import { registerGscTools } from "./gscTools.js";
import { registerGa4Tools } from "./ga4Tools.js";

export function registerAllTools(server) {
  registerAccountTools(server);
  registerGscTools(server);
  registerGa4Tools(server);
}
