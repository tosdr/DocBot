import { Regex } from '../models';

module.exports = {
    	expression: new RegExp("^((?=.*without signing up for)|(?=.*without registering an account)|(?=.*not require you to register)|(?=.*no user registration)|((?=.*don’t need an account)|(?=.*do not need an account)))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 320,
	name: "No need to register"
} as Regex;