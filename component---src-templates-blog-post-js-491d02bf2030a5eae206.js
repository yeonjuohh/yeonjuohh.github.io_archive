(self.webpackChunk=self.webpackChunk||[]).push([[989],{6179:function(t,e,n){"use strict";var r=n(7294),a=n(5414),i=n(5444),o=function(t){var e,n,o,c=t.description,u=t.lang,f=t.meta,l=t.title,s=(0,i.useStaticQuery)("2841359383").site,p=c||s.siteMetadata.description,d=null===(e=s.siteMetadata)||void 0===e?void 0:e.title;return r.createElement(a.q,{htmlAttributes:{lang:u},title:l,titleTemplate:d?"%s | "+d:null,meta:[{name:"description",content:p},{property:"og:title",content:l},{property:"og:description",content:p},{property:"og:type",content:"website"},{name:"twitter:card",content:"summary"},{name:"twitter:creator",content:(null===(n=s.siteMetadata)||void 0===n||null===(o=n.social)||void 0===o?void 0:o.twitter)||""},{name:"twitter:title",content:l},{name:"twitter:description",content:p}].concat(f)})};o.defaultProps={lang:"en",meta:[],description:""},e.Z=o},2024:function(t,e,n){"use strict";n.r(e),n.d(e,{default:function(){return s}});var r=n(7294),a=n(5444),i=n(1804),o=n.n(i),c=n(6125),u=function(){return r.createElement("div",{className:"bio"},r.createElement("div",{className:"bio_pic"},r.createElement(c.S,{className:"bio-avatar",src:"../images/profile-pic.png",width:65,height:65,alt:"Profile picture",__imageData:n(120)})),r.createElement("div",null,r.createElement(a.Link,{to:"/about"},r.createElement("strong",null,"Hado")),r.createElement("br",null),"세상에 관심이 많은 데이터 분석가입니다. ",r.createElement("br",null)))},f=n(7198),l=n(6179);n(749);var s=function(t){var e,n=t.data,i=t.location,c=n.markdownRemark,s=(null===(e=n.site.siteMetadata)||void 0===e?void 0:e.title)||"Title",p=n.previous,d=n.next;return r.createElement(f.Z,{location:i,title:s},r.createElement(l.Z,{title:c.frontmatter.title,description:c.frontmatter.description||c.excerpt}),r.createElement("article",{className:"blog-post",itemScope:!0,itemType:"http://schema.org/Article"},r.createElement("div",{className:"blog-post-header"},r.createElement("h1",{className:"blog-post-title"},c.frontmatter.title),r.createElement("div",{className:"blog-post-date"},c.frontmatter.date),c.frontmatter.tags.map((function(t){return r.createElement(a.Link,{to:"/tags/"+o()(t)+"/"},r.createElement("div",{className:"blog-post-category"}," ",t," "))}))),r.createElement("section",{dangerouslySetInnerHTML:{__html:c.html},itemProp:"articleBody"}),r.createElement("footer",null,r.createElement(u,null))),r.createElement("nav",{className:"blog-post-nav"},r.createElement("ul",{style:{display:"flex",flexWrap:"wrap",justifyContent:"space-between",listStyle:"none",padding:0}},r.createElement("li",null,p&&r.createElement(a.Link,{to:p.fields.slug,rel:"prev"},"← ",p.frontmatter.title)),r.createElement("li",null,d&&r.createElement(a.Link,{to:d.fields.slug,rel:"next"},d.frontmatter.title," →")))))}},2705:function(t,e,n){var r=n(5639).Symbol;t.exports=r},9932:function(t){t.exports=function(t,e){for(var n=-1,r=null==t?0:t.length,a=Array(r);++n<r;)a[n]=e(t[n],n,t);return a}},2663:function(t){t.exports=function(t,e,n,r){var a=-1,i=null==t?0:t.length;for(r&&i&&(n=t[++a]);++a<i;)n=e(n,t[a],a,t);return n}},9029:function(t){var e=/[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g;t.exports=function(t){return t.match(e)||[]}},4239:function(t,e,n){var r=n(2705),a=n(9607),i=n(2333),o=r?r.toStringTag:void 0;t.exports=function(t){return null==t?void 0===t?"[object Undefined]":"[object Null]":o&&o in Object(t)?a(t):i(t)}},8674:function(t){t.exports=function(t){return function(e){return null==t?void 0:t[e]}}},531:function(t,e,n){var r=n(2705),a=n(9932),i=n(1469),o=n(3448),c=r?r.prototype:void 0,u=c?c.toString:void 0;t.exports=function t(e){if("string"==typeof e)return e;if(i(e))return a(e,t)+"";if(o(e))return u?u.call(e):"";var n=e+"";return"0"==n&&1/e==-Infinity?"-0":n}},5393:function(t,e,n){var r=n(2663),a=n(3816),i=n(8748),o=RegExp("['’]","g");t.exports=function(t){return function(e){return r(i(a(e).replace(o,"")),t,"")}}},9389:function(t,e,n){var r=n(8674)({"À":"A","Á":"A","Â":"A","Ã":"A","Ä":"A","Å":"A","à":"a","á":"a","â":"a","ã":"a","ä":"a","å":"a","Ç":"C","ç":"c","Ð":"D","ð":"d","È":"E","É":"E","Ê":"E","Ë":"E","è":"e","é":"e","ê":"e","ë":"e","Ì":"I","Í":"I","Î":"I","Ï":"I","ì":"i","í":"i","î":"i","ï":"i","Ñ":"N","ñ":"n","Ò":"O","Ó":"O","Ô":"O","Õ":"O","Ö":"O","Ø":"O","ò":"o","ó":"o","ô":"o","õ":"o","ö":"o","ø":"o","Ù":"U","Ú":"U","Û":"U","Ü":"U","ù":"u","ú":"u","û":"u","ü":"u","Ý":"Y","ý":"y","ÿ":"y","Æ":"Ae","æ":"ae","Þ":"Th","þ":"th","ß":"ss","Ā":"A","Ă":"A","Ą":"A","ā":"a","ă":"a","ą":"a","Ć":"C","Ĉ":"C","Ċ":"C","Č":"C","ć":"c","ĉ":"c","ċ":"c","č":"c","Ď":"D","Đ":"D","ď":"d","đ":"d","Ē":"E","Ĕ":"E","Ė":"E","Ę":"E","Ě":"E","ē":"e","ĕ":"e","ė":"e","ę":"e","ě":"e","Ĝ":"G","Ğ":"G","Ġ":"G","Ģ":"G","ĝ":"g","ğ":"g","ġ":"g","ģ":"g","Ĥ":"H","Ħ":"H","ĥ":"h","ħ":"h","Ĩ":"I","Ī":"I","Ĭ":"I","Į":"I","İ":"I","ĩ":"i","ī":"i","ĭ":"i","į":"i","ı":"i","Ĵ":"J","ĵ":"j","Ķ":"K","ķ":"k","ĸ":"k","Ĺ":"L","Ļ":"L","Ľ":"L","Ŀ":"L","Ł":"L","ĺ":"l","ļ":"l","ľ":"l","ŀ":"l","ł":"l","Ń":"N","Ņ":"N","Ň":"N","Ŋ":"N","ń":"n","ņ":"n","ň":"n","ŋ":"n","Ō":"O","Ŏ":"O","Ő":"O","ō":"o","ŏ":"o","ő":"o","Ŕ":"R","Ŗ":"R","Ř":"R","ŕ":"r","ŗ":"r","ř":"r","Ś":"S","Ŝ":"S","Ş":"S","Š":"S","ś":"s","ŝ":"s","ş":"s","š":"s","Ţ":"T","Ť":"T","Ŧ":"T","ţ":"t","ť":"t","ŧ":"t","Ũ":"U","Ū":"U","Ŭ":"U","Ů":"U","Ű":"U","Ų":"U","ũ":"u","ū":"u","ŭ":"u","ů":"u","ű":"u","ų":"u","Ŵ":"W","ŵ":"w","Ŷ":"Y","ŷ":"y","Ÿ":"Y","Ź":"Z","Ż":"Z","Ž":"Z","ź":"z","ż":"z","ž":"z","Ĳ":"IJ","ĳ":"ij","Œ":"Oe","œ":"oe","ŉ":"'n","ſ":"s"});t.exports=r},1957:function(t,e,n){var r="object"==typeof n.g&&n.g&&n.g.Object===Object&&n.g;t.exports=r},9607:function(t,e,n){var r=n(2705),a=Object.prototype,i=a.hasOwnProperty,o=a.toString,c=r?r.toStringTag:void 0;t.exports=function(t){var e=i.call(t,c),n=t[c];try{t[c]=void 0;var r=!0}catch(u){}var a=o.call(t);return r&&(e?t[c]=n:delete t[c]),a}},3157:function(t){var e=/[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/;t.exports=function(t){return e.test(t)}},2333:function(t){var e=Object.prototype.toString;t.exports=function(t){return e.call(t)}},5639:function(t,e,n){var r=n(1957),a="object"==typeof self&&self&&self.Object===Object&&self,i=r||a||Function("return this")();t.exports=i},2757:function(t){var e="\\u2700-\\u27bf",n="a-z\\xdf-\\xf6\\xf8-\\xff",r="A-Z\\xc0-\\xd6\\xd8-\\xde",a="\\xac\\xb1\\xd7\\xf7\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf\\u2000-\\u206f \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000",i="["+a+"]",o="\\d+",c="[\\u2700-\\u27bf]",u="["+n+"]",f="[^\\ud800-\\udfff"+a+o+e+n+r+"]",l="(?:\\ud83c[\\udde6-\\uddff]){2}",s="[\\ud800-\\udbff][\\udc00-\\udfff]",p="["+r+"]",d="(?:"+u+"|"+f+")",m="(?:"+p+"|"+f+")",b="(?:['’](?:d|ll|m|re|s|t|ve))?",x="(?:['’](?:D|LL|M|RE|S|T|VE))?",g="(?:[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]|\\ud83c[\\udffb-\\udfff])?",v="[\\ufe0e\\ufe0f]?",E=v+g+("(?:\\u200d(?:"+["[^\\ud800-\\udfff]",l,s].join("|")+")"+v+g+")*"),y="(?:"+[c,l,s].join("|")+")"+E,w=RegExp([p+"?"+u+"+"+b+"(?="+[i,p,"$"].join("|")+")",m+"+"+x+"(?="+[i,p+d,"$"].join("|")+")",p+"?"+d+"+"+b,p+"+"+x,"\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])","\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])",o,y].join("|"),"g");t.exports=function(t){return t.match(w)||[]}},3816:function(t,e,n){var r=n(9389),a=n(9833),i=/[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g,o=RegExp("[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]","g");t.exports=function(t){return(t=a(t))&&t.replace(i,r).replace(o,"")}},1469:function(t){var e=Array.isArray;t.exports=e},7005:function(t){t.exports=function(t){return null!=t&&"object"==typeof t}},3448:function(t,e,n){var r=n(4239),a=n(7005);t.exports=function(t){return"symbol"==typeof t||a(t)&&"[object Symbol]"==r(t)}},1804:function(t,e,n){var r=n(5393)((function(t,e,n){return t+(n?"-":"")+e.toLowerCase()}));t.exports=r},9833:function(t,e,n){var r=n(531);t.exports=function(t){return null==t?"":r(t)}},8748:function(t,e,n){var r=n(9029),a=n(3157),i=n(9833),o=n(2757);t.exports=function(t,e,n){return t=i(t),void 0===(e=n?void 0:e)?a(t)?o(t):r(t):t.match(e)||[]}},749:function(t,e,n){"use strict";n.r(e)},120:function(t){"use strict";t.exports=JSON.parse('{"layout":"constrained","backgroundColor":"#f8f8f8","images":{"fallback":{"src":"/static/f2ede77506a2004bf59054f108a1077b/b024c/profile-pic.png","srcSet":"/static/f2ede77506a2004bf59054f108a1077b/fbc98/profile-pic.png 16w,\\n/static/f2ede77506a2004bf59054f108a1077b/f9f53/profile-pic.png 33w,\\n/static/f2ede77506a2004bf59054f108a1077b/b024c/profile-pic.png 65w,\\n/static/f2ede77506a2004bf59054f108a1077b/c1559/profile-pic.png 130w","sizes":"(min-width: 65px) 65px, 100vw"},"sources":[{"srcSet":"/static/f2ede77506a2004bf59054f108a1077b/e789a/profile-pic.webp 16w,\\n/static/f2ede77506a2004bf59054f108a1077b/0cc22/profile-pic.webp 33w,\\n/static/f2ede77506a2004bf59054f108a1077b/0c531/profile-pic.webp 65w,\\n/static/f2ede77506a2004bf59054f108a1077b/a819e/profile-pic.webp 130w","type":"image/webp","sizes":"(min-width: 65px) 65px, 100vw"}]},"width":65,"height":65}')}}]);
//# sourceMappingURL=component---src-templates-blog-post-js-491d02bf2030a5eae206.js.map