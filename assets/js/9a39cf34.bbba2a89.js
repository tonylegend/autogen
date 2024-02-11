"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8096],{8488:(e,c,n)=>{n.r(c),n.d(c,{assets:()=>d,contentTitle:()=>i,default:()=>u,frontMatter:()=>r,metadata:()=>s,toc:()=>a});var t=n(5893),o=n(1151);const r={sidebar_label:"factory",title:"coding.factory"},i=void 0,s={id:"reference/coding/factory",title:"coding.factory",description:"CodeExecutorFactory Objects",source:"@site/docs/reference/coding/factory.md",sourceDirName:"reference/coding",slug:"/reference/coding/factory",permalink:"/autogen/docs/reference/coding/factory",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/autogen/edit/main/website/docs/reference/coding/factory.md",tags:[],version:"current",frontMatter:{sidebar_label:"factory",title:"coding.factory"},sidebar:"referenceSideBar",previous:{title:"embedded_ipython_code_executor",permalink:"/autogen/docs/reference/coding/embedded_ipython_code_executor"},next:{title:"local_commandline_code_executor",permalink:"/autogen/docs/reference/coding/local_commandline_code_executor"}},d={},a=[{value:"CodeExecutorFactory Objects",id:"codeexecutorfactory-objects",level:2},{value:"create",id:"create",level:4}];function l(e){const c={code:"code",em:"em",h2:"h2",h4:"h4",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,o.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(c.h2,{id:"codeexecutorfactory-objects",children:"CodeExecutorFactory Objects"}),"\n",(0,t.jsx)(c.pre,{children:(0,t.jsx)(c.code,{className:"language-python",children:"class CodeExecutorFactory()\n"})}),"\n",(0,t.jsx)(c.p,{children:"(Experimental) A factory class for creating code executors."}),"\n",(0,t.jsx)(c.h4,{id:"create",children:"create"}),"\n",(0,t.jsx)(c.pre,{children:(0,t.jsx)(c.code,{className:"language-python",children:"@staticmethod\ndef create(code_execution_config: Dict[str, Any]) -> CodeExecutor\n"})}),"\n",(0,t.jsx)(c.p,{children:"(Experimental) Get a code executor based on the code execution config."}),"\n",(0,t.jsxs)(c.p,{children:[(0,t.jsx)(c.strong,{children:"Arguments"}),":"]}),"\n",(0,t.jsxs)(c.ul,{children:["\n",(0,t.jsxs)(c.li,{children:[(0,t.jsx)(c.code,{children:"code_execution_config"})," ",(0,t.jsx)(c.em,{children:"Dict"}),' - The code execution config,\nwhich is a dictionary that must contain the key "executor".\nThe value of the key "executor" can be either a string\nor an instance of CodeExecutor, in which case the code\nexecutor is returned directly.']}),"\n"]}),"\n",(0,t.jsxs)(c.p,{children:[(0,t.jsx)(c.strong,{children:"Returns"}),":"]}),"\n",(0,t.jsxs)(c.ul,{children:["\n",(0,t.jsxs)(c.li,{children:[(0,t.jsx)(c.code,{children:"CodeExecutor"})," - The code executor."]}),"\n"]}),"\n",(0,t.jsxs)(c.p,{children:[(0,t.jsx)(c.strong,{children:"Raises"}),":"]}),"\n",(0,t.jsxs)(c.ul,{children:["\n",(0,t.jsxs)(c.li,{children:[(0,t.jsx)(c.code,{children:"ValueError"})," - If the code executor is unknown or not specified."]}),"\n"]})]})}function u(e={}){const{wrapper:c}={...(0,o.a)(),...e.components};return c?(0,t.jsx)(c,{...e,children:(0,t.jsx)(l,{...e})}):l(e)}},1151:(e,c,n)=>{n.d(c,{Z:()=>s,a:()=>i});var t=n(7294);const o={},r=t.createContext(o);function i(e){const c=t.useContext(r);return t.useMemo((function(){return"function"==typeof e?e(c):{...c,...e}}),[c,e])}function s(e){let c;return c=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:i(e.components),t.createElement(r.Provider,{value:c},e.children)}}}]);