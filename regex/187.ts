import { Regex } from '../models';

module.exports = {
     	expression: new RegExp("^((((?=.*transfer)(?=.*outside)((?=.*country)|(?=.*countries)|(?=.*EEA)))|(?=.*transfer)((?=.*around the world)|(?=.*around the globe)))|((?=.*pd)|(?=.*your personal)((?=.*data)|(?=.*info)))(?=.*other)((?=.*countries)|(?=.*country)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 187,
	name: "Your data may be processed and stored anywhere in the world"
} as Regex;