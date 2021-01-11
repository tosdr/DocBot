import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*Facebook)(?=.*pixel))|(?=.*social)((?=.*network)|((?=.*media)))(?=.*cookie))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 307,
	name: "The service uses social media cookies/pixels"
} as Regex;