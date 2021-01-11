import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*opt out)|(?=.*opt-out)|(?=.*unsubscribing)|(?=.*unsubscribe))(((?=.*email)|(?=.*promotion))|(?=.*market)(?=.*communications))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 223,
	name: "You can opt out of promotional communications"
} as Regex;