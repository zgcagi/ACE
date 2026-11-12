"""Regressions for migrating the remaining React/TSX substring scorers.

These tests intentionally use the production ``score_probe`` dispatch path.  The
old keyword bags, comment/string stuffing, and one semantic near-miss per
probe must fail, while the reference answers and representative identifier-
renamed equivalents must pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE))

from eval.dispatch import load_policy, score_probe  # noqa: E402


POLICY = load_policy(ACE / "policy" / "atom_intents.yaml")


def _load_code_probes() -> dict[str, dict]:
    probes = {}
    for raw in (ACE / "probes" / "code.jsonl").read_text().splitlines():
        if raw.lstrip().startswith("{"):
            probe = json.loads(raw)
            probes[probe["probe_id"]] = probe
    return probes


PROBES = _load_code_probes()

OLD_SUBSTRINGS = {
    "code.component_authoring.L1.001": ["Greeting", "name: string", "<h1>", "Hello", "{name}", "</h1>"],
    "code.layout_composition.L1.001": ["display:", "flex", "justifyContent", "alignItems", "Home", "About", "Contact"],
    "code.layout_composition.L1.002": ["display:", "flex", "flexDirection", "column", "gap", "children"],
    "code.layout_composition.L1.004": ["display:", "flex", "justifyContent", "flex-end"],
    "code.layout_composition.L1.005": ["className", "flex", "justify-center", "w-full", "children"],
    "code.layout_composition.L2.001": ["display:", "grid", "gridTemplateColumns", "repeat", "1fr", "gap"],
    "code.layout_composition.L2.003": ["grid", "grid-cols-1", "md:grid-cols-2", "lg:grid-cols-4", "gap-4"],
    "code.layout_composition.L2.004": ["display:", "grid", "gridTemplateColumns", "auto-fill", "minmax", "200px", "1fr", "gap"],
    "code.layout_composition.L3.001": ["display:", "grid", "gridTemplateColumns", "240px", "1fr", "gridTemplateRows", "60px", "sticky", "overflow"],
    "code.layout_composition.L3.002": ["flex", "flex-col", "h-screen", "h-12", "w-64", "flex-1", "overflow-y-auto"],
    "code.state_management.L2.005": ["useState", "setItems", "reduce", "total", "map", "onClick"],
    "code.state_management.L3.001": ["useEffect", "useState", "setInterval", "clearInterval", "return ()=>"],
    "code.state_management.L3.002": ["useEffect", "useState", "addEventListener", "removeEventListener", "resize", "return ()=>"],
    "code.state_management.L3.003": ["useEffect", "useState", "setTimeout", "clearTimeout", "AbortController", "abort", "fetch", "debounced"],
    "code.state_management.L3.005": ["useEffect", "useState", "setInterval", "clearInterval", "visibilitychange", "addEventListener", "removeEventListener", "fetch"],
    "code.component_authoring.L1.002": ["Avatar", "src", "alt", "<img"],
    "code.component_authoring.L1.003": ["Card", "title", "body", "<h2>", "<p>", "</div>"],
    "code.component_authoring.L1.004": ["Banner", "banner", "Welcome aboard!", "<div"],
    "code.component_authoring.L1.005": ["Spacer", "size", "style", "height"],
    "code.component_authoring.L2.003": ["function colorFor", "Alert", "severity", "background", "children"],
    "code.component_authoring.L3.004": ["TreeNode", "function Node", "TreeView", "children", "<ul>", "<li>", "map"],
    "code.component_authoring.L3.005": ["function THead", "function TRow", "function Pager", "DataTable", "columns", "rows", "pageSize", "map", "Math.ceil"],
    "code.import_resolution.L1.003": ["import", "useState", "useEffect", "from 'react'"],
    "code.import_resolution.L2.001": ["import", "LineChart", "XAxis", "YAxis", "Tooltip", "Line", "from 'recharts'"],
    "code.import_resolution.L2.003": ["Box", "Button", "@mui/material", "Link", "Routes", "Route", "react-router-dom"],
    "code.import_resolution.L2.005": ["orderBy", "lodash", "startOfDay", "format", "date-fns"],
    "code.import_resolution.L3.001": ["import React", "useState", "useEffect", "from 'react'", "import axios from 'axios'", "formatNumber", "capitalize", "./helpers"],
    "code.import_resolution.L3.002": ["ThemeProvider", "@mui/material/styles", "CssBaseline", "BrowserRouter as Router", "Routes", "Route", "react-router-dom", "./theme", "./Home"],
    "code.import_resolution.L3.004": ["useState", "import _ from 'lodash'", "import cn from 'classnames'", "Helper", "Tag", "./Helper"],
}


EQUIVALENT_VARIANTS = {
    "code.layout_composition.L1.001": """\
export function Header(){return <nav style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}><a href="#">Home</a><a href="#">About</a><a href="#">Contact</a></nav>}""",
    "code.state_management.L2.005": """\
import { useState } from "react";
type Item = { id: string; qty: number; price: number };
export function Cart() {
  const [cart, updateCart] = useState<Item[]>([]);
  const sum = cart.reduce((acc, product) => acc + product.qty * product.price, 0);
  const addProduct = () => updateCart(previous => {
    const found = previous.find(product => product.id === "a");
    return found
      ? previous.map(product => product.id === "a"
          ? { ...product, qty: product.qty + 1 } : product)
      : [...previous, { id: "a", qty: 1, price: 5 }];
  });
  const deleteProduct = (wanted: string) =>
    updateCart(previous => previous.filter(product => product.id !== wanted));
  return <div><span>{sum}</span><button onClick={addProduct}>Add a</button>
    {cart.map(product => <button key={product.id}
      onClick={() => deleteProduct(product.id)}>Remove</button>)}</div>;
}""",
    "code.state_management.L3.001": """\
import { useEffect, useState } from "react";
export function Clock() {
  const [moment, updateMoment] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => updateMoment(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const hours = String(moment.getHours()).padStart(2, "0");
  const minutes = String(moment.getMinutes()).padStart(2, "0");
  const seconds = String(moment.getSeconds()).padStart(2, "0");
  return <time>{hours}:{minutes}:{seconds}</time>;
}""",
    "code.state_management.L3.002": """\
import { useEffect, useState } from "react";
export function WindowSize() {
  const [size, updateSize] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    function handleResize() {
      updateSize({ w: window.innerWidth, h: window.innerHeight });
    }
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return <output>{size.w} x {size.h}</output>;
}""",
    "code.state_management.L3.003": """\
import { useEffect, useState } from "react";
export function Search() {
  const [text, setText] = useState("");
  const [settled, setSettled] = useState("");
  const [matches, setMatches] = useState<any[]>([]);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(text), 300);
    return () => clearTimeout(timer);
  }, [text]);
  useEffect(() => {
    if (!settled) return;
    const controller = new AbortController();
    fetch("/api?q=" + settled, { signal: controller.signal })
      .then(response => response.json()).then(setMatches);
    return () => controller.abort();
  }, [settled]);
  return <><input value={text} onChange={event => setText(event.target.value)} />
    {matches.map((match, index) => <div key={index}>{match}</div>)}</>;
}""",
    "code.state_management.L3.005": """\
import { useEffect, useState } from "react";
export function Poller() {
  const [latest, setLatest] = useState("");
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    const request = () => fetch("/api/status").then(reply => reply.text()).then(setLatest);
    const begin = () => { if (timer === null) timer = setInterval(request, 5000); };
    const end = () => { if (timer !== null) { clearInterval(timer); timer = null; } };
    const visibility = () => { if (document.hidden) end(); else begin(); };
    document.addEventListener("visibilitychange", visibility);
    visibility();
    return () => { end(); document.removeEventListener("visibilitychange", visibility); };
  }, []);
  return <output>{latest}</output>;
}""",
    "code.component_authoring.L3.004": """\
import type { ReactNode } from "react";
export type TreeNode<T> = { value: T; children: TreeNode<T>[] };
function Node<T>({ node: branch, render: show }: { node: TreeNode<T>; render: (value: T) => ReactNode }) {
  return <li>{show(branch.value)}{branch.children.length > 0 &&
    <ul>{branch.children.map((descendant, position) =>
      <Node key={position} node={descendant} render={show} />)}</ul>}</li>;
}
export function TreeView<T>({ root, render }: { root: TreeNode<T>; render: (value: T) => ReactNode }) {
  return <ul><Node node={root} render={render} /></ul>;
}""",
    "code.component_authoring.L3.005": """\
function THead<T>({ cols }: { cols: { key: keyof T; label: string }[] }) {
  return <thead><tr>{cols.map(column => <th key={String(column.key)}>{column.label}</th>)}</tr></thead>;
}
function TRow<T>({ row: record, cols }: { row: T; cols: { key: keyof T; label: string }[] }) {
  return <tr>{cols.map(column => <td key={String(column.key)}>{String(record[column.key])}</td>)}</tr>;
}
function Pager({ page, total }: { page: number; total: number }) {
  return <div>page {page}/{total}</div>;
}
export function DataTable<T>({ columns, rows, pageSize }: {
  columns: { key: keyof T; label: string }[]; rows: T[]; pageSize: number;
}) {
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const pageRows = rows.slice(0, pageSize);
  return <div><table><THead cols={columns} /><tbody>{pageRows.map((record, position) =>
    <TRow key={position} row={record} cols={columns} />)}</tbody></table>
    <Pager page={1} total={pages} /></div>;
}""",
    "code.import_resolution.L1.003": "import { useMemo, useEffect, useState } from \"react\";",
    "code.import_resolution.L2.001": "import { Tooltip, Line, YAxis, LineChart, XAxis } from \"recharts\";",
    "code.import_resolution.L2.003": "import { Button, Box } from \"@mui/material\";\nimport { Route, Link, Routes } from \"react-router-dom\";",
    "code.import_resolution.L2.005": "import { orderBy } from \"lodash\";\nimport { format, startOfDay } from \"date-fns\";",
    "code.import_resolution.L3.001": "import React, { useEffect, useState } from \"react\";\nimport axios from \"axios\";\nimport { capitalize, formatNumber } from \"./helpers\";",
    "code.import_resolution.L3.004": "import { useState } from \"react\";\nimport _ from \"lodash\";\nimport cn from \"classnames\";\nimport Helper, { Tag } from \"./Helper\";",
}


SEMANTIC_NEAR_MISSES = {
    "code.component_authoring.L1.001": """\
export function Greeting({name}:{name:string}){return <h1>Goodbye, {name}!</h1>}""",
    "code.layout_composition.L1.001": """\
export function Header(){return <nav style={{display:"flex",justifyContent:"center",alignItems:"center"}}><a href="#">Home</a><a href="#">About</a><a href="#">Contact</a></nav>}""",
    "code.layout_composition.L1.002": """\
export function VStack({children}:{children:React.ReactNode}){return <div style={{display:"flex",flexDirection:"row",alignItems:"flex-start",gap:8}}>{children}</div>}""",
    "code.layout_composition.L1.004": """\
export function Toolbar({children}:{children:React.ReactNode}){return <div style={{display:"flex",justifyContent:"flex-start"}}>{children}</div>}""",
    "code.layout_composition.L1.005": """\
export function Centered({children}:{children:React.ReactNode}){return <div className="flex w-auto justify-center">{children}</div>}""",
    "code.layout_composition.L2.001": """\
export function CardGrid({children}:{children:React.ReactNode}){return <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:16}}>{children}</div>}""",
    "code.layout_composition.L2.003": """\
export function TileGrid({children}:{children:React.ReactNode}){return <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">{children}</div>}""",
    "code.layout_composition.L2.004": """\
export function Photos({children}:{children:React.ReactNode}){return <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(100px,1fr))",gap:12}}>{children}</div>}""",
    "code.layout_composition.L3.001": """\
export function AppShell({topbar,sidebar,main}:any){return <div style={{display:"grid",gridTemplateColumns:"240px minmax(0,1fr)",gridTemplateRows:"60px minmax(0,1fr)",height:"100vh"}}><header style={{gridColumn:"1 / -1",position:"sticky",top:0}}>{topbar}</header><aside style={{overflowY:"auto"}}>{sidebar}</aside><main style={{minHeight:0}}>{main}</main></div>}""",
    "code.layout_composition.L3.002": """\
export function ChatLayout({header,sidebar,messages,input}:any){return <div className="flex h-screen flex-col"><header className="h-12 shrink-0">{header}</header><div className="flex min-h-0 flex-1"><aside className="w-64 shrink-0">{sidebar}</aside><main className="flex min-h-0 flex-1 flex-col"><div className="min-h-0 flex-1">{messages}</div><div className="overflow-y-auto shrink-0">{input}</div></main></div></div>}""",
    "code.state_management.L2.005": """\
import {useState} from "react"; type Item={id:string;qty:number;price:number};
export function Cart(){const [items,setItems]=useState<Item[]>([]);
const total=items.reduce((s,i)=>s+i.qty*i.price,0);
const add=()=>setItems(xs=>[...xs,{id:"a",qty:1,price:5}]);
const remove=(id:string)=>setItems(xs=>xs);
return <div>{total}<button onClick={add}>Add a</button>
{items.map(i=><button onClick={()=>remove(i.id)}>Remove</button>)}</div>}""",
    "code.state_management.L3.001": """\
import {useEffect,useState} from "react"; export function Clock(){const [time,setTime]=useState(()=>new Date());
useEffect(()=>{const id=setInterval(()=>setTime(new Date()),1000);return()=>clearInterval(other)},[]);
return <div>{String(time.getHours()).padStart(2,"0")}:{String(time.getMinutes()).padStart(2,"0")}:{String(time.getSeconds()).padStart(2,"0")}</div>}""",
    "code.state_management.L3.002": """\
import {useEffect,useState} from "react"; export function WindowSize(){const [size,setSize]=useState({w:window.innerWidth,h:window.innerHeight});
useEffect(()=>{const resize=()=>setSize({w:window.innerWidth,h:window.innerHeight});window.addEventListener("resize",resize);return()=>window.removeEventListener("resize",()=>resize())},[]);
return <div>{size.w} x {size.h}</div>}""",
    "code.state_management.L3.003": """\
import {useEffect,useState} from "react"; export function Search(){const [query,setQuery]=useState("");const [debounced,setDebounced]=useState("");const [results,setResults]=useState([]);
useEffect(()=>{const id=setTimeout(()=>setDebounced(query),300);return()=>clearTimeout(id)},[query]);
useEffect(()=>{if(!debounced)return;const ac=new AbortController();fetch("/api?q="+debounced).then(r=>r.json()).then(setResults);return()=>{}},[debounced]);
return <><input value={query} onChange={e=>setQuery(e.target.value)}/>{results.map(x=><div>{x}</div>)}</>}""",
    "code.state_management.L3.005": """\
import {useEffect,useState} from "react"; export function Poller(){const [status,setStatus]=useState("");
useEffect(()=>{const id=setInterval(()=>fetch("/api/status").then(r=>r.text()).then(setStatus),5000);
const visible=()=>{};document.addEventListener("visibilitychange",visible);
return()=>{clearInterval(id);document.removeEventListener("visibilitychange",visible)}},[]);
return <div>{status}</div>}""",
    "code.component_authoring.L1.002": """\
export function Avatar({src,alt}:{src:string;alt:string}){return <img src={alt} alt={src}/>}""",
    "code.component_authoring.L1.003": """\
export function Card({title,body}:{title:string;body:string}){return <div><h2>{body}</h2><p>{title}</p></div>}""",
    "code.component_authoring.L1.004": """\
export function Banner(){return <div className="banner">Welcome back!</div>}""",
    "code.component_authoring.L1.005": """\
export function Spacer({size}:{size:number}){return <div style={{height:size+1}}/>}""",
    "code.component_authoring.L2.003": """\
import type {ReactNode} from "react"; function colorFor(severity:"info"|"warn"|"error"){switch(severity){case "info":return "blue";case "warn":return "yellow";case "error":return "red"}}
export function Alert({severity,children}:{severity:"info"|"warn"|"error";children:ReactNode}){return <div style={{background:"white"}}>{children}</div>}""",
    "code.component_authoring.L3.004": """\
import type {ReactNode} from "react"; export type TreeNode<T>={value:T;children:TreeNode<T>[]};
function Node<T>({node,render}:{node:TreeNode<T>;render:(value:T)=>ReactNode}){return <li>{render(node.value)}{node.children.length>0&&<ul>{node.children.map(child=><li>{render(child.value)}</li>)}</ul>}</li>}
export function TreeView<T>({root,render}:{root:TreeNode<T>;render:(value:T)=>ReactNode}){return <ul><Node node={root} render={render}/></ul>}""",
    "code.component_authoring.L3.005": """\
function THead<T>({cols}:any){return <thead><tr>{cols.map((c:any)=><th>{c.label}</th>)}</tr></thead>}
function TRow<T>({row,cols}:any){return <tr>{cols.map((c:any)=><td>{String(row[c.key])}</td>)}</tr>}
function Pager({page,total}:any){return <div>page {page}/{total}</div>}
export function DataTable<T>({columns,rows,pageSize}:any){const total=Math.max(1,Math.ceil(rows.length/pageSize));return <div><table><THead cols={columns}/><tbody>{rows.map((row:any)=><TRow row={row} cols={columns}/>)}</tbody></table><Pager page={1} total={total}/></div>}""",
    "code.import_resolution.L1.003": """\
import { useState, useMemo } from "react";""",
    "code.import_resolution.L2.001": """\
import { LineChart, XAxis, YAxis, Tooltip } from "recharts";""",
    "code.import_resolution.L2.003": """\
import { Box } from "@mui/material"; import { Button, Link, Routes, Route } from "react-router-dom";""",
    "code.import_resolution.L2.005": """\
import orderBy from "lodash"; import { startOfDay, format } from "date-fns";""",
    "code.import_resolution.L3.001": """\
import { useState, useEffect } from "react"; import axios from "axios"; import { formatNumber, capitalize } from "./helpers";""",
    "code.import_resolution.L3.002": """\
import { ThemeProvider } from "@mui/material/styles"; import CssBaseline from "@mui/material/CssBaseline";
import { BrowserRouter, Routes, Route } from "react-router-dom"; import theme from "./theme"; import Home from "./Home";""",
    "code.import_resolution.L3.004": """\
import { useState } from "react"; import { uniq as _ } from "lodash"; import cn from "classnames"; import Helper, { Tag } from "./Helper";""",
}


COMPONENT_TARGETS = {
    "code.component_authoring.L1.001": "Greeting",
    "code.layout_composition.L1.001": "Header",
    "code.layout_composition.L1.002": "VStack",
    "code.layout_composition.L1.004": "Toolbar",
    "code.layout_composition.L1.005": "Centered",
    "code.layout_composition.L2.001": "CardGrid",
    "code.layout_composition.L2.003": "TileGrid",
    "code.layout_composition.L2.004": "Photos",
    "code.layout_composition.L3.001": "AppShell",
    "code.layout_composition.L3.002": "ChatLayout",
    "code.component_authoring.L1.002": "Avatar",
    "code.component_authoring.L1.003": "Card",
    "code.component_authoring.L1.004": "Banner",
    "code.component_authoring.L1.005": "Spacer",
    "code.component_authoring.L2.003": "Alert",
    "code.component_authoring.L3.004": "TreeView",
    "code.component_authoring.L3.005": "DataTable",
}


ADDITIONAL_SEMANTIC_NEAR_MISSES = {
    "greeting_correct_jsx_only_in_string": (
        "code.component_authoring.L1.001",
        """function Greeting({name}:{name:string}) {
          const bait = "<h1>Hello, {name}!</h1>";
          return null;
        }""",
    ),
    "greeting_correct_jsx_only_in_comment": (
        "code.component_authoring.L1.001",
        """function Greeting({name}:{name:string}) {
          /* return <h1>Hello, {name}!</h1>; */
          return null;
        }""",
    ),
    "header_one_dead_correct_return_then_null": (
        "code.layout_composition.L1.001",
        """function Header(){
          if (false) return <nav style={{display:"flex",justifyContent:"space-between",
            alignItems:"center"}}><a href="#">Home</a><a href="#">About</a>
            <a href="#">Contact</a></nav>;
          return null;
        }""",
    ),
    "header_constant_dead_conditional": (
        "code.layout_composition.L1.001",
        """function Header(){return false
          ? <nav style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <a href="#">Home</a><a href="#">About</a><a href="#">Contact</a>
            </nav>
          : null;
        }""",
    ),
    "toolbar_wrong_prop_type": (
        "code.layout_composition.L1.004",
        """export function Toolbar({children}:{children:string}) {
          return <div style={{display:"flex",justifyContent:"flex-end"}}>{children}</div>;
        }""",
    ),
    "header_invalid_statement_syntax": (
        "code.layout_composition.L1.001",
        """function Header(){ totally invalid tokens here;
          return <nav style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <a href="#">Home</a><a href="#">About</a><a href="#">Contact</a></nav>;
        }""",
    ),
    "alert_colors_not_distinct": (
        "code.component_authoring.L2.003",
        """import type {ReactNode} from "react";
        function colorFor(severity:"info"|"warn"|"error") {
          switch(severity) {
            case "info": return "red";
            case "warn": return "red";
            case "error": return "red";
          }
        }
        export function Alert({severity,children}:{severity:"info"|"warn"|"error";children:ReactNode}) {
          return <div style={{background:colorFor(severity)}}>{children}</div>;
        }""",
    ),
    "photos_empty_grid": (
        "code.layout_composition.L2.004",
        """export function Photos(){return <div style={{display:"grid",
          gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:12}}></div>}""",
    ),
    "photos_null_only_grid_child": (
        "code.layout_composition.L2.004",
        """export function Photos(){return <div style={{display:"grid",
          gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:12}}>
          {null}</div>}""",
    ),
    "tree_exported_nongeneric_any": (
        "code.component_authoring.L3.004",
        """import type {ReactNode} from "react";
        export type TreeNode<T>={value:T;children:TreeNode<T>[]};
        export function Node({node,render}:any){return <li>{render(node.value)}
          {node.children.length>0&&<ul>{node.children.map((child:any)=>
          <Node node={child} render={render}/>)}</ul>}</li>}
        export function TreeView({root,render}:any){
          return <ul><Node node={root} render={render}/></ul>
        }""",
    ),
    "datatable_nongeneric_any": (
        "code.component_authoring.L3.005",
        """function THead({cols}:any){return <thead><tr>{cols.map((c:any)=>
          <th>{c.label}</th>)}</tr></thead>}
        function TRow({row,cols}:any){return <tr>{cols.map((c:any)=>
          <td>{String(row[c.key])}</td>)}</tr>}
        function Pager({page,total}:any){return <div>page {page}/{total}</div>}
        export function DataTable({columns,rows,pageSize}:any){
          const total=Math.max(1,Math.ceil(rows.length/pageSize));
          const visible=rows.slice(0,pageSize);
          return <><table><THead cols={columns}/><tbody>{visible.map((row:any)=>
          <TRow row={row} cols={columns}/>)}</tbody></table>
          <Pager page={1} total={total}/></>
        }""",
    ),
}


def _score(probe_id: str, response: str):
    return score_probe(PROBES[probe_id], response, POLICY)


def test_exactly_the_reviewed_29_probes_use_tsx_static_contracts() -> None:
    assert len(OLD_SUBSTRINGS) == 29
    assert set(SEMANTIC_NEAR_MISSES) == set(OLD_SUBSTRINGS)
    for probe_id in OLD_SUBSTRINGS:
        scorer = PROBES[probe_id]["scorer"]
        assert scorer["type"] == "code_parse_ast", (probe_id, scorer)
        assert scorer["parser"] == "tsx", (probe_id, scorer)
        assert scorer.get("checks"), probe_id


def test_all_reference_answers_pass_through_production_dispatch() -> None:
    failures = []
    for probe_id in OLD_SUBSTRINGS:
        result, denominator = _score(probe_id, PROBES[probe_id]["ground_truth"])
        if denominator != 1 or not result.passed:
            failures.append((probe_id, denominator, result.to_dict()))
    assert not failures, failures


def test_all_enabled_deterministic_code_ground_truths_pass() -> None:
    failures = []
    for probe_id, probe in PROBES.items():
        if not probe.get("enabled", True):
            result, denominator = _score(probe_id, probe.get("ground_truth", ""))
            if denominator != 0 or not result.deferred:
                failures.append((probe_id, denominator, result.to_dict()))
            continue
        if probe["scorer"].get("type") == "judge_rubric":
            # Runtime Judge availability is covered by the Judge integration
            # suite; this test is the local deterministic ground-truth gate.
            continue
        result, denominator = _score(probe_id, probe.get("ground_truth", ""))
        if denominator != 1 or not result.passed:
            failures.append((probe_id, denominator, result.to_dict()))
    assert not failures, failures


def test_identifier_renames_and_import_reordering_remain_accepted() -> None:
    assert len(EQUIVALENT_VARIANTS) >= 10
    failures = []
    for probe_id, response in EQUIVALENT_VARIANTS.items():
        result, denominator = _score(probe_id, response)
        if denominator != 1 or not result.passed:
            failures.append((probe_id, denominator, result.to_dict()))
    assert not failures, failures


def test_every_old_keyword_bag_and_noncode_stuffing_fails() -> None:
    failures = []
    for probe_id, substrings in OLD_SUBSTRINGS.items():
        old_false_positive = " ".join(substrings)
        variants = (
            old_false_positive,
            f"/* {old_false_positive} */",
            f"const bait = {json.dumps(old_false_positive)};",
        )
        for label, response in zip(("bare", "comment", "string"), variants):
            result, denominator = _score(probe_id, response)
            if denominator != 1 or result.passed:
                failures.append((probe_id, label, denominator, result.to_dict()))
    assert not failures, failures


def test_every_probe_rejects_a_semantic_near_miss() -> None:
    failures = []
    for probe_id, response in SEMANTIC_NEAR_MISSES.items():
        result, denominator = _score(probe_id, response)
        if denominator != 1 or result.passed:
            failures.append((probe_id, denominator, result.to_dict()))
    assert not failures, failures


def test_component_contracts_bind_requirements_to_the_named_component() -> None:
    failures = []
    for probe_id, component in COMPONENT_TARGETS.items():
        ground_truth = PROBES[probe_id]["ground_truth"]
        dead = ground_truth.replace(
            f"function {component}", f"function Dead{component}", 1,
        )
        assert dead != ground_truth, probe_id
        response = dead + f"\nexport function {component}() {{ return null; }}\n"
        result, denominator = _score(probe_id, response)
        if denominator != 1 or result.passed:
            failures.append((probe_id, denominator, result.to_dict()))
    assert not failures, failures


def test_additional_type_syntax_and_behavior_near_misses_fail() -> None:
    failures = []
    for label, (probe_id, response) in ADDITIONAL_SEMANTIC_NEAR_MISSES.items():
        result, denominator = _score(probe_id, response)
        if denominator != 1 or result.passed:
            failures.append((label, probe_id, denominator, result.to_dict()))
    assert not failures, failures
