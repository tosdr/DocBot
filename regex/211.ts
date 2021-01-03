import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*you)(?=.*info)((?=.*promotion)|(?=.*sweepstake)|(?=.*contest)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 211,
	name: "The service may collect extra data about you through promotions"
} as Regex;